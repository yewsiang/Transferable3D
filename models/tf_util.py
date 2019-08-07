
""" Utility functions for Point Cloud, 3D Box and BoxPC manipulation.

Author: Tang Yew Siang
Date: July 2019
"""

import numpy as np
import tensorflow as tf

def tf_retrieve_masked_points_and_centroids(point_cloud, mask, target_points):
  """
  Select points from a point_cloud (B,N,D) according to the 1s in a binary mask (B,N)
  and resample these selected points so that they will be of target_points long (B,T,D).

  Since there are variable number of 1s in the mask (N_seg_points), the selected points
  will be of [(N_seg_points_1, D), ..., (N_seg_points_B, D)] where N_seg_points_i is different
  from each other and dependent on the mask.
  
  We resample each (N_seg_points_i,D) so that it becomes (T,D). Combining the B batches of
  (T,D) points, we get (B,T,D).

  Inputs:
      point_cloud: (B, N, D)
      mask: (B, N) Binary mask.
      target_points: Integer specifying the number of points per batch for the output.
  Outputs:
      final_points: (B, T, D)
      centroids: (B, N) Centroids of x,y,z values of point_cloud. The other dimensions
          are set to zero. 
  """
  batch_size, num_points = mask.get_shape()

  with tf.variable_scope('retrieve_masked_points_and_centroids'):

    # Point Cloud is (B, N, 4). For each of the points (N, 4) of the B batches,
    # We split it into 2 different groups (one group being the group with 1s in mask &
    # the other being the group with 0s) using tf.dynamic_partition
    groups = 2
    # ids (B, N) are masks with {0, 1} for every B.
    # We convert it into a format such that every row is unique in order to partition
    # them into (B x Groups) unique partitions.
    # Eg. From [[0,0,0,1], [1,1,0,1], [0,1,1,0]] -> [[0,0,0,1], [3,3,2,3], [4,5,5,4]]
    # For B = 1: There will be 3 in Group 0, 1 in Group 1.
    # For B = 3: There will be 2 in Group 4, 2 in Group 5.
    unique_ids = tf.transpose(mask, [1,0]) + groups * tf.range(0, batch_size)
    unique_ids = tf.transpose(unique_ids, [1,0])
    groups = tf.dynamic_partition(point_cloud, unique_ids, batch_size * groups)

    # For each of the points (N, 4) of the B batches, it has been split into 2 groups
    # (1 group being the non-selected points and the other being the mask)
    # We need to resample the mask so that it reaches the target number of points
    # (Eg. from 279 -> 512)
    final_points = []
    centroids = []
    for b in range(batch_size):
      # From the above, we explained the the masks are split into unique ids.
      # Eg. From [[0,0,0,1], [1,1,0,1], [0,1,1,0]] -> [[0,0,0,1], [3,3,2,3], [4,5,5,4]]
      # Notice that the masked points are given the ids (row number * 2 + 1)
      idx = b * 2 + 1

      # Check if mask is empty
      N_seg_points = tf.shape(groups[idx])[0]
      have_segmented_points = tf.greater(N_seg_points, 0)
      # Resample from the current points until we have target points (if not empty).
      # If empty, we give dummy values of 0s every where
      resampled_points = tf.cond(have_segmented_points, 
                                 true_fn=lambda : tf_resample_points(groups[idx], target_points), 
                                 false_fn=lambda : tf.zeros([target_points, 4]))
      final_points.append(resampled_points)

      # Calculate centroid value from points (if not empty).
      # If empty, we give dummy values of 0s
      centroid = tf.cond(have_segmented_points, 
                         true_fn=lambda : tf_get_xyz_centroid(groups[idx]), 
                         false_fn=lambda : tf.zeros([4]))
      centroids.append(centroid)

    final_points = tf.stack(final_points, axis=0)
    centroids = tf.stack(centroids, axis=0)

  return final_points, centroids

def tf_resample_point_clouds(point_clouds, target_points, N_points=None):
  """
  Resamples point clouds (B, N, 4) such that it is of target_points (T) long and
  becomes (B, T, 4)
  """
  with tf.variable_scope('resample_point_clouds'):
  
    resampled_pcs = tf.map_fn(lambda pc: tf_resample_points(pc, target_points, N_points=N_points), point_clouds)
  
  return resampled_pcs

def tf_resample_points(point_cloud, target_points, N_points=None):
  """
  Resample from point_cloud (N, D) such that it is of target_points (T) long and
  becomes (T, D).

  If N >= T, then T points will be sampled without replacement from N.
  If N < T, then the 1st N points will remain the same and the remaining (T-N) resampled
  from the point_cloud.
  Inputs:
      point_cloud: (N, D)
      target_points: Number of points to reach, T. point_cloud will become (T, D)
      N_points: Number of points (N) of the current point cloud.
  Outputs:
      resampled: (T, D)
  """

  if N_points is None:
    N_points = tf.shape(point_cloud)[0]

  def tf_sample_without_replacement(point_cloud, target_points):
    permutation = tf.random_shuffle(tf.range(0, N_points))
    chosen = tf.gather(permutation, tf.range(0, target_points))
    sampled = tf.gather(point_cloud, chosen)
    return sampled

  def tf_sample_remaining(point_cloud, target_points):
    chosen = tf.random_uniform([target_points - N_points], minval=0, maxval=N_points, dtype=tf.int32)
    newly_sampled = tf.gather(point_cloud, chosen)
    resampled = tf.concat([point_cloud, newly_sampled], axis=0)
    return resampled

  with tf.variable_scope('resample_points'):
     
    have_enough_points = tf.greater_equal(N_points, target_points)
    sampled = tf.cond(have_enough_points, true_fn=lambda : tf_sample_without_replacement(point_cloud, target_points),
                                          false_fn=lambda : tf_sample_remaining(point_cloud, target_points))
  
  return sampled

def tf_normalize_point_clouds_to_01(point_clouds):
  """
  Normalizes a point cloud along the x,y,z dimensions to range [0,1] and mean 0. 
  Last dimensions of point clouds are kept untouched.
  """
  B, N, D = point_clouds.get_shape().as_list()

  with tf.variable_scope('normalize_point_clouds_to_01'):

    centroids = tf.reduce_mean(point_clouds, axis=1)
    centroids_exp = tf.expand_dims(centroids, axis=1)
    centroids_exp = tf.tile(centroids_exp,[1,N,1])
    translated_pc = point_clouds - centroids_exp

    max_dims = tf.reduce_max(translated_pc, axis=1) - tf.reduce_min(translated_pc, axis=1)
    max_dims = tf.reduce_max(max_dims[:,:3], axis=1)
    max_dims_exp = tf.tile(tf.reshape(max_dims, [B,1,1]), [1,N,D])

    normalized_pc = translated_pc / (max_dims_exp + 1e-5)
    normalized_pc = tf.concat([normalized_pc[:,:,:3], point_clouds[:,:,3:]], axis=2)

  return normalized_pc

def tf_normalize_point_clouds_to_mean_zero_and_unit_var(point_clouds):
  """
  Normalizes a point cloud along the x,y,z dimensions to mean 0 and unit variance.
  Last dimensions of point clouds are kept untouched.
  """
  B, N, D = point_clouds.get_shape().as_list()

  with tf.variable_scope('normalize_point_clouds_to_mean_zero_and_unit_var'):

    centroids, variances = tf.nn.moments(point_clouds, axes=[1])
    centroids_exp = tf.tile(tf.expand_dims(centroids, axis=1), [1,N,1])
    deviation_exp = tf.tile(tf.expand_dims(tf.sqrt(variances), axis=1), [1,N,1])

    normalized_pc = (point_clouds - centroids_exp) / (deviation_exp + 1e-5)
    normalized_pc = tf.concat([normalized_pc[:,:,:3], point_clouds[:,:,3:]], axis=2)

  return normalized_pc

def tf_reduce_mask_max(point_cloud_features, mask):
  """
  Given a point cloud (B,N,F) and a mask (B,N), use the mask to weight the features from the N points,
  this are the masked features.
  Then, choose the max indices using argmax and use the indices to select from the original features
  instead of from the masked features to preserve the original statistic of the max operation.
  """
  B, N, F = point_cloud_features.get_shape().as_list()

  with tf.variable_scope('reduce_mask_max'):

    mask_exp = tf_expand_tile(mask, axis=2, tile=[1,1,F])            # (B,N,F)
    masked_pc_features = point_cloud_features * (mask_exp + 1e-5)    # (B,N,F)
    masked_pc_features_T = tf.transpose(masked_pc_features, [0,2,1]) # (B,F,N)
    pc_features_T = tf.transpose(point_cloud_features, [0,2,1])      # (B,F,N)
    selected_feats = tf.argmax(masked_pc_features_T, axis=2)         # (B,F)

    # Filler ids
    ids1 = tf.tile(tf.reshape(tf.range(B, dtype=tf.int64), [B,1]), [1,F]) # (B,F)
    ids2 = tf.tile(tf.reshape(tf.range(F, dtype=tf.int64), [1,F]), [B,1]) # (B,F)

    # Final ID. Final dimension is 3 since Rank 3 for masked_pc_features_T
    ids = tf.stack([ids1, ids2, selected_feats], axis=2) # (B,F,3)
    unmasked_features = tf.gather_nd(pc_features_T, ids) # (B,F)

  return unmasked_features


def tf_get_xyz_centroid(point_cloud):
  """
  Given a point_cloud of (N, 3 + D) points, (Eg. N points with each point (x, y, z, ref) (D = 1)),
  we calculate the centroid using only the 1st three dimensions (N, 3) and set the last dimension 
  to 0 since we only want to xyz centroid.
  Inputs:
      point_cloud: (N, 3 + D)
  Outputs:
      centroid_xyz_with_d: (3 + D,) where 1st 3 dimensions are the mean of the (N, 3) points and
          the remaining (D,) is set to 0.
  """

  with tf.variable_scope('get_xyz_centroid'):

    centroid = tf.reduce_mean(point_cloud, axis=0)
    centroid_xyz = tf.gather(centroid, [0,1,2])
    zeros_for_d = tf.zeros([tf.shape(point_cloud)[1] - 3])
    centroid_xyz_with_d = tf.concat([centroid_xyz, zeros_for_d], axis=0)

  return centroid_xyz_with_d

def tf_get_xyz_centroid_and_variance_of_point_clouds(point_clouds):

  with tf.variable_scope('get_xyz_centroid_and_variance_of_point_clouds'):

    centroids = tf.map_fn(lambda pc: tf_get_xyz_centroid_and_variance(pc)[0], point_clouds)
    variances = tf.map_fn(lambda pc: tf_get_xyz_centroid_and_variance(pc)[1], point_clouds)

  return centroids, variances

def tf_get_xyz_centroid_and_variance(point_cloud, dims=4):

  with tf.variable_scope('get_xyz_centroid_and_variance'):

    centroid, variance = tf.nn.moments(point_cloud, axes=[1])
    centroid_xyz = tf.gather(centroid, [0,1,2])
    variance_xyz = tf.gather(variance, [0,1,2])
    #zeros_for_d = tf.zeros([tf.shape(point_cloud)[1] - 3])
    zeros_for_d = tf.zeros([dims - 3])

    centroid_xyz_with_d = tf.concat([centroid_xyz, zeros_for_d], axis=0)
    variance_xyz_with_d = tf.concat([variance_xyz, zeros_for_d], axis=0)

  return centroid_xyz_with_d, variance_xyz_with_d

def tf_get_soft_xyz_centroids_and_variances(point_clouds, soft_masks):
  """
  Weights the centroid & variance contributed by each point according to the weight on the soft mask.
  Note: This is for multiple point_clouds (B,N,3+D) and multiple soft_masks (B,N)
  """
  xyz = point_clouds
  B, N, D = point_clouds.get_shape().as_list()

  with tf.variable_scope('soft_centroids_and_variances'):

    # Normalize the mask
    mask_sum = tf.reduce_sum(soft_masks, axis=1) + 1e-5
    soft_masks = tf.transpose(soft_masks, [1,0]) / mask_sum
    soft_masks = tf.transpose(soft_masks, [1,0])
    soft_mask_exp = tf.tile(tf.expand_dims(soft_masks, axis=2), [1,1,D])

    # Weight each point according to its soft mask
    weighted_point_clouds = point_clouds * soft_mask_exp
    weighted_centroids = tf.reduce_sum(weighted_point_clouds, axis=1)
    weighted_centroids_exp = tf.expand_dims(weighted_centroids, axis=1)
    weighted_centroids_exp = tf.tile(weighted_centroids_exp,[1,N,1])

    # Weight variance contribution from each point according to its soft mask
    variance = tf.square(point_clouds - weighted_centroids_exp)
    # variance = tf.square(weighted_point_clouds - weighted_centroids_exp)
    weighted_variance = variance * soft_mask_exp
    variances = tf.reduce_sum(weighted_variance, axis=1)

    # Add the other dimensions back in
    zeros_for_d = tf.zeros([B,D - 3])
    centroid_xyz = tf.slice(weighted_centroids, [0,0], [-1,3])
    variance_xyz = tf.slice(variances, [0,0], [-1,3])
    centroid_xyz_with_d = tf.concat([centroid_xyz, zeros_for_d], axis=1)
    variance_xyz_with_d = tf.concat([variance_xyz, zeros_for_d], axis=1)

  return centroid_xyz_with_d, variance_xyz_with_d

def tf_separate_points_into_separate_classes_using_mask(point_cloud, mask, cls_id, target_points):
  """
  Most code (with comments) from tf_retrieve_masked_points_and_centroids (above).
  """
  batch_size, num_points = mask.get_shape()

  with tf.variable_scope('separate_points_into_separate_classes_using_mask'):

    groups = 2
    unique_ids = tf.transpose(mask, [1,0]) + groups * tf.range(0, batch_size)
    unique_ids = tf.transpose(unique_ids, [1,0])
    groups = tf.dynamic_partition(point_cloud, unique_ids, batch_size * groups)

    # For each of the points (N, 4) of the B batches, it has been split into 2 groups
    # (1 group being the non-selected points and the other being the mask)
    # We need to resample the mask so that it reaches the target number of points
    # (Eg. from 279 -> 512)
    separated_points_list = []
    cls_labels = []
    centroids = []
    variances = []
    for b in range(batch_size):
      # From the above, we explained the the masks are split into unique ids.
      # Eg. From [[0,0,0,1], [1,1,0,1], [0,1,1,0]] -> [[0,0,0,1], [3,3,2,3], [4,5,5,4]]
      # Notice that the masked points are given the ids (row number * 2 + 1)
      idx = b * 2

      # Background points
      separated_points, centroid, _ = tf_resample_points_with_checks(groups[idx], target_points)
      # Set background variance to 0 to prevent penalization
      variance = tf.zeros([4])
      separated_points_list.append(separated_points)
      cls_labels.append(0)
      centroids.append(centroid)
      variances.append(variance)

      # Masked points
      separated_points, centroid, variance = tf_resample_points_with_checks(groups[idx + 1], target_points)
      separated_points_list.append(separated_points)
      cls_labels.append(tf.gather(cls_id, b) + 1)
      centroids.append(centroid)
      variances.append(variance)
      """
      resampled_points = tf_resample_points(tf.gather(point_cloud, b), target_points)
      separated_points_list.append(resampled_points)
      cls_labels.append(tf.gather(cls_id, b) + 1 + 3)
      centroid, variance = tf_get_xyz_centroid_and_variance(resampled_points)
      centroids.append(centroid)
      variances.append(variance)
      """

    separated_points_list = tf.stack(separated_points_list, axis=0)
    cls_labels = tf.stack(cls_labels, axis=0)
    centroids = tf.stack(centroids, axis=0)
    variances = tf.stack(variances, axis=0)

  return separated_points, cls_labels, centroids, variances

def tf_resample_points_with_checks(point_cloud, target_points):

  with tf.variable_scope('resample_points_with_checks'):

    # Check if mask is empty
    N_seg_points = tf.shape(point_cloud)[0]
    have_segmented_points = tf.greater(N_seg_points, 0)
    # Resample from the current points until we have target points (if not empty).
    # If empty, we give dummy values of 0s every where
    resampled_points = tf.cond(have_segmented_points, 
                               true_fn=lambda : tf_resample_points(point_cloud, target_points), 
                               false_fn=lambda : tf.zeros([target_points, 4]))
    
    # Calculate centroid and variance value from points (if not empty).
    # If empty, we give dummy values of 0s
    centroid, variance = tf.cond(have_segmented_points, 
                                 true_fn=lambda : tf_get_xyz_centroid_and_variance(point_cloud), 
                                 false_fn=lambda : (tf.zeros([4]), tf.zeros([4])) )

  return resampled_points, centroid, variance

def tf_get_2D_bbox_of_points(points2D):
  """ 
  Get the 2D bbox of points2D, assuming it represents 2D image coords.
  Inputs: 
    points2D (N,2)
  """
  with tf.variable_scope('tf_get_2D_bbox_of_points'):

    left   = tf.reduce_min(points2D[:,0])
    top    = tf.reduce_min(points2D[:,1])
    right  = tf.reduce_max(points2D[:,0])
    bottom = tf.reduce_max(points2D[:,1])

  return [left, top, right, bottom]

def tf_get_2D_softmax_bbox_of_points(points2D, softmax_scale_factor):
  """ 
  Get the 2D bbox of points2D, assuming it represents 2D image coords.
  Instead of taking maximum, we use softmax.
  Inputs: 
    points2D (N,2)
  """
  with tf.variable_scope('tf_get_2D_softmax_bbox_of_points'):

    left_bound   = tf.reduce_min(points2D[:,0])
    top_bound    = tf.reduce_min(points2D[:,1])
    right_bound  = tf.reduce_max(points2D[:,0])
    bottom_bound = tf.reduce_max(points2D[:,1])
    width  = tf.stop_gradient(tf.abs(right_bound - left_bound))
    height = tf.stop_gradient(tf.abs(bottom_bound - top_bound))

    # left_closeness is much you are to the left when compared to the right etc
    # For a point that is at the right side, it would be 0 to the left
    # For a point that is at the left side, it would a certain amt to the left
    left_closeness   = right_bound - points2D[:,0]  # (N,) 
    top_closeness    = bottom_bound - points2D[:,1] # (N,)
    right_closeness  = points2D[:,0] - left_bound   # (N,)
    bottom_closeness = points2D[:,1] - top_bound    # (N,)

    left_softmax   = tf.nn.softmax((left_closeness / width) * softmax_scale_factor)    # (N,) 
    top_softmax    = tf.nn.softmax((top_closeness / height) * softmax_scale_factor)    # (N,) 
    right_softmax  = tf.nn.softmax((right_closeness / width) * softmax_scale_factor)   # (N,) 
    bottom_softmax = tf.nn.softmax((bottom_closeness / height) * softmax_scale_factor) # (N,) 

    left   = tf.reduce_sum(tf.multiply(points2D[:,0], tf.stop_gradient(left_softmax)))
    top    = tf.reduce_sum(tf.multiply(points2D[:,1], tf.stop_gradient(top_softmax)))
    right  = tf.reduce_sum(tf.multiply(points2D[:,0], tf.stop_gradient(right_softmax)))
    bottom = tf.reduce_sum(tf.multiply(points2D[:,1], tf.stop_gradient(bottom_softmax)))

  return [left, top, right, bottom]

# SUNRGBD
def tf_get_2D_bbox_of_projection_sunrgbd_multi(point_clouds, Rtilts, Ks):
  """ 
  Given a point cloud, project it onto the image plane and get the 2D bbox that covers exactly
  the projected points (take maximum).
  Inputs: (B,N,3), (B,3,3), (B,3,3)
  Output: (B,4)
  """
  with tf.variable_scope('get_2D_bbox_of_projection_sunrgbd_multi'):

    point_clouds = project_upright_camera_to_upright_depth(point_clouds)
    projected_points, _ = project_upright_depth_to_image(point_clouds, Rtilts, Ks) # (B,N,2)
    projected_bboxes = tf.map_fn(lambda x: tf_get_2D_bbox_of_points(x[0]), 
                                 [projected_points], dtype=[tf.float32, tf.float32, tf.float32, tf.float32])
    projected_bboxes = tf.stack(projected_bboxes, axis=1)

  return projected_bboxes

# SUNRGBD
def tf_get_2D_bbox_of_softmax_projection_sunrgbd_multi(point_clouds, Rtilts, Ks, softmax_scale_factor):
  """ 
  Given a point cloud, project it onto the image plane and get the 2D bbox that is the softmax of
  the projected points.
  Inputs: (B,N,3), (B,3,3), (B,3,3), ()
  Output: (B,4)
  """
  with tf.variable_scope('get_2D_bbox_of_softmax_projection_sunrgbd_multi'):

    point_clouds = project_upright_camera_to_upright_depth(point_clouds)
    projected_points, _ = project_upright_depth_to_image(point_clouds, Rtilts, Ks) # (B,8,2)
    projected_bboxes = tf.map_fn(lambda x: tf_get_2D_softmax_bbox_of_points(x[0], softmax_scale_factor), 
                                 [projected_points], dtype=[tf.float32, tf.float32, tf.float32, tf.float32])
    projected_bboxes = tf.stack(projected_bboxes, axis=1)

  return projected_bboxes

# SUNRGBD
def tf_get_projected_points_sunrgbd_multi(point_clouds, Rtilts, Ks):
  """ 
  Given a point cloud, project it onto the image plane and get the 2D bbox that covers exactly
  the projected points.
  Inputs: (B,N,3), (B,3,3), (B,3,3)
  Output: (B,N,2)
  """
  with tf.variable_scope('get_projected_points_sunrgbd_multi'):

    point_clouds = project_upright_camera_to_upright_depth(point_clouds)
    projected_points, _ = project_upright_depth_to_image(point_clouds, Rtilts, Ks) # (B,N,2)

  return projected_points

def tf_normalize_2D_bboxes(box2D, image_dim):
  """
  Given a list of 2D bboxes, get the normalized coordinates which lies in
  [0,1] by dividing by rows or cols of the entire image.
  Inputs:
    box2D: (B,4)
    image_dim: (B,2)
  """
  with tf.variable_scope('clip_2D_bbox_to_image_dims'):

    rows = image_dim[:,0] # (B,)
    cols = image_dim[:,1] # (B,)
    left   = box2D[:,0] / cols # (B,)
    top    = box2D[:,1] / rows # (B,)
    right  = box2D[:,2] / cols # (B,)
    bottom = box2D[:,3] / rows # (B,)
    new_box2D = tf.stack([left, top, right, bottom], axis=1) # (B,4)
  
  return new_box2D

def tf_dilate_2D_bboxes(bbox2D, dilate_factor):
  """
  Given a list of 2D bboxes, dilate the height and width of the bboxes.
  2D bboxes should be given in [left, top, right, bottom] format
  Inputs:
    bbox2D: (B,4)
    dilate_factor: ()
  Output:
    new_bbox2D: (B,4)
  """
  left   = bbox2D[:,0] # (B,)
  top    = bbox2D[:,1] # (B,)
  right  = bbox2D[:,2] # (B,)
  bottom = bbox2D[:,3] # (B,)
  center_x = (left + right) / 2.
  center_y = (top + bottom) / 2.
  width  = right - left
  height = top - bottom # TODO: Should flip bottom & top but doesn't really matter
  tf.assert_greater(width, 0.)
  tf.assert_greater(height, 0.)

  new_width  = dilate_factor * width
  new_height = dilate_factor * height
  new_left   = center_x - (new_width / 2.)
  new_right  = center_x + (new_width / 2.)
  new_bottom = center_y - (new_height / 2.) # TODO
  new_top    = center_y + (new_height / 2.) # TODO
  new_bbox2D = tf.stack([new_left, new_top, new_right, new_bottom], axis=1)
  return new_bbox2D

# SUNRGBD (Because of rows, cols)
def tf_clip_2D_bbox_to_image_dims(box2D, image_dim):

  with tf.variable_scope('clip_2D_bbox_to_image_dims'):

    img_rows = image_dim[0]
    img_cols = image_dim[1]
    col_min = tf.maximum(0., box2D[0])
    row_min = tf.maximum(0., box2D[1])
    col_max = tf.minimum(img_cols, box2D[2])
    row_max = tf.minimum(img_rows, box2D[3])
  
  return [col_min, row_min, col_max, row_max]

# SUNRGBD (Because of rows, cols)
def tf_clip_2D_bbox_to_image_dims_multi(box2Ds, image_dims):

  with tf.variable_scope('clip_2D_bbox_to_image_dims_multi'):

    clipped_box2Ds = tf.map_fn(lambda x: tf_clip_2D_bbox_to_image_dims(x[0], x[1]), 
                               [box2Ds, image_dims], 
                               dtype=[tf.float32, tf.float32, tf.float32, tf.float32])
    clipped_box2Ds = tf.stack(clipped_box2Ds, axis=1)
    
  return clipped_box2Ds

def tf_check_if_points_lie_within_box(points, box_params):
  """
  Checks if points lie within the box or on the surfaces of a given box and not outside.
  Note: Instead of the more intuitive (N,3) dims for points, it is (N,6,3).

  Inputs:
    points    : (N,6,3) A (3,) vector for every surface.
    box_params: ((3,), (3,), ())
  Outputs:
    are_points_inside_box: (N,6)
  """
  N, _, _ = points.get_shape().as_list()

  # Corners (3,). Axes (3',3)
  corners, axes = tf_create_3D_box_by_corner_and_axes(box_params)
  corners_exp = tf.tile(tf.reshape(corners, [1,1,3]), [N,6,1]) # (N,6,3)
  axes_exp = tf.tile(tf.reshape(axes, [1,1,3,3]), [N,6,1,1])   # (N,6,3',3)
  length_of_surfaces = tf.norm(axes_exp, ord=2, axis=3)        # (N,6,3')

  points_ray = points - corners_exp                                   # (N,6,3)
  points_ray_exp = tf_expand_tile(points_ray, axis=2, tile=[1,1,3,1]) # (N,6,3',3)
  points_length_along_surfaces = tf.reduce_sum(points_ray_exp * axes_exp, axis=3)           # (N,6,3')
  points_length_along_surfaces = points_length_along_surfaces / (length_of_surfaces + 1e-5) # (N,6,3')
  points_extra_length_along_surfaces = length_of_surfaces - points_length_along_surfaces    # (N,6,3')

  # Check that the lengths of the points along surfaces is bounded between [0, length_of_surfaces]
  points_in_right_direction = tf.greater_equal((points_length_along_surfaces + 1e-5), 0)       # (N,6,3')
  points_within_dimensions  = tf.greater_equal((points_extra_length_along_surfaces + 1e-5), 0) # (N,6,3')
  points_in_right_direction_for_all_surfaces = tf.reduce_prod(tf.cast(points_in_right_direction, tf.int32), axis=2) # (N,6)
  points_within_dimensions_for_all_surfaces  = tf.reduce_prod(tf.cast(points_within_dimensions, tf.int32), axis=2)  # (N,6)
  are_points_inside_box = points_in_right_direction_for_all_surfaces * points_within_dimensions_for_all_surfaces    # (N,6)

  return are_points_inside_box

def tf_check_if_point_cloud_lie_within_box(points, box_params):
  """
  Checks if points lie within the box or on the surfaces of a given box and not outside.
  Note: Same as tf_check_if_points_lie_within_box but on points with (N,3) instead of (N,6,3).

  Inputs:
    points    : (N,3)
    box_params: ((3,), (3,), ())
  Outputs:
    are_points_inside_box: (N,6)
  """
  N, _ = points.get_shape().as_list()

  # Corners (3,). Axes (3',3)
  corners, axes = tf_create_3D_box_by_corner_and_axes(box_params)
  corners_exp = tf.tile(tf.reshape(corners, [1,3]), [N,1]) # (N,3)
  axes_exp = tf.tile(tf.reshape(axes, [1,3,3]), [N,1,1])   # (N,3',3)
  length_of_surfaces = tf.norm(axes_exp, ord=2, axis=2)    # (N,3')

  points_ray = points - corners_exp                                 # (N,3)
  points_ray_exp = tf_expand_tile(points_ray, axis=1, tile=[1,3,1]) # (N,3',3)
  points_length_along_surfaces = tf.reduce_sum(points_ray_exp * axes_exp, axis=2)           # (N,3')
  points_length_along_surfaces = points_length_along_surfaces / (length_of_surfaces + 1e-5) # (N,3')
  points_extra_length_along_surfaces = length_of_surfaces - points_length_along_surfaces    # (N,3')

  # Check that the lengths of the points along surfaces is bounded between [0, length_of_surfaces]
  points_in_right_direction = tf.greater_equal((points_length_along_surfaces + 1e-5), 0)       # (N,3')
  points_within_dimensions  = tf.greater_equal((points_extra_length_along_surfaces + 1e-5), 0) # (N,3')
  points_in_right_direction_for_all_surfaces = tf.reduce_prod(tf.cast(points_in_right_direction, tf.int32), axis=1) # (N,)
  points_within_dimensions_for_all_surfaces  = tf.reduce_prod(tf.cast(points_within_dimensions, tf.int32), axis=1)  # (N,)
  are_points_inside_box = points_in_right_direction_for_all_surfaces * points_within_dimensions_for_all_surfaces    # (N,)

  return are_points_inside_box

def tf_distance_to_box_surfaces(point_cloud, box_params):
  """
  Calculate the distance from each point in the point_cloud to the 6 surfaces of the 3D box along
  the ray that passes through the center of the box.
  
  Distance from box center to the intersection point at the box surface, d = (p0 - l0).n / l.n 
  where p0 is a point on the plane, l0 is box center, l is the ray from box center to intersection
  and n is the normal vectors of the surfaces.
  To get the interesection: dl + l0
  (https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection)

  Inputs:
    point_cloud: (N,3)
    box_params : ((3,), (3,), ())
  Outputs:
    intersection_pts       : (N,6,3) Intersection point betw ray (box center to point) & surface planes.
      These intersection points may fall outside of the box volume (so have to check for this).
    dist_points_to_surfaces: (N,6) Distance from point to surface plane.
  """

  with tf.variable_scope('distance_to_box_surfaces'):

    N, _ = point_cloud.get_shape().as_list()
    # Center l0 (3,), Surface points (6,3), Surface normals (6,3)
    center, surface_pts, surface_norms = tf_create_3D_box_by_surface_centers(box_params,
                                                                             apply_translation=True,
                                                                             use_base=False)

    ray_to_points = point_cloud - center # (N,3)
    ray_to_points_exp = tf.tile(tf.expand_dims(ray_to_points, axis=1), [1,6,1]) # l (N,6,3) 
    surface_norms_exp = tf.tile(tf.expand_dims(surface_norms, axis=0), [N,1,1]) # n (N,6,3)

    # Perpendicular distance (l.n). NOTE: Might have zeros if no intersection.
    perp_distance = ray_to_points_exp * surface_norms_exp # (N,6,3)
    perp_distance = tf.reduce_sum(perp_distance, axis=2)  # (N,6)

    # (p0 - l0).n
    p0_minus_l0 = surface_pts - center # (6,3)
    p0_minus_l0_dot_n = tf.reduce_sum(p0_minus_l0 * surface_norms, axis=1)            # (6,)
    p0_minus_l0_dot_n_exp = tf.tile(tf.expand_dims(p0_minus_l0_dot_n, axis=0), [N,1]) # (N,6)

    # Distance from center to surface (d times l). d = (p0 - l0).n / l.n
    # NOTE: Possibility of inf if the center-point ray is parallel to some surfaces 
    # i.e (l.n or perpendicular distance = 0)
    dist_center_to_surfaces = p0_minus_l0_dot_n_exp / (perp_distance + 1e-5) # (N,6)
    dist_center_to_surfaces = tf.norm(ray_to_points_exp, ord=2, axis=2) * dist_center_to_surfaces # (N,6)

    # Check if point is indeed at center
    # If there are points that are at the center, it will cause problems. 
    # Hence, we replace 0s with a small value.
    l, w, h = box_params[1][0], box_params[1][1], box_params[1][2]
    dist_center_to_surfaces = tf.where(tf.equal(tf.reduce_sum(tf.abs(ray_to_points), axis=1), 0),
                                       tf_expand_tile(tf.convert_to_tensor([l/2,l/2,h/2,h/2,w/2,w/2]), axis=0, tile=[N,1]), 
                                       dist_center_to_surfaces)

    # Distance from points to surface
    dist_center_to_points   = tf.norm(ray_to_points_exp, ord=2, axis=2) # (N,6)
    dist_points_to_surfaces = tf.abs(dist_center_to_points - dist_center_to_surfaces) # (N,6)

    # Intersection point at the surfaces
    dist_center_to_points_exp   = tf.tile(tf.expand_dims(dist_center_to_points, axis=2), [1,1,3])   # (N,6,3)
    dist_center_to_surfaces_exp = tf.tile(tf.expand_dims(dist_center_to_surfaces, axis=2), [1,1,3]) # (N,6,3)
    ray_to_points_norm_exp = ray_to_points_exp / (dist_center_to_points_exp + 1e-5) # (N,6,3)
    ray_to_intersection = dist_center_to_surfaces_exp * ray_to_points_norm_exp      # (N,6,3)
    center_exp = tf.tile(tf.reshape(center, [1,1,3]), [N,6,1]) # (N,6,3)
    intersection_pts = ray_to_intersection + center_exp        # dl + l0 (N,6,3)

  return intersection_pts, dist_points_to_surfaces

def tf_distance_to_closest_3D_box_surface(point_cloud, box_params):
  """
  Inputs:
    point_cloud: (N,3)
    box_params : ((3,), (3,), (2,))
  Outputs:
    min_dist_points_to_surfaces: (N,)
  """

  with tf.variable_scope('distance_to_closest_3D_box_surface'):

    # Intersection points (N,6,3). Distance from points to surfaces (N,6)
    intersection_pts, dist_points_to_surfaces = tf_distance_to_box_surfaces(point_cloud, box_params)

    # Integer array rep whether points are inside the box or not (N,6)
    are_points_inside_box = tf_check_if_points_lie_within_box(intersection_pts, box_params)

    # Set distances to a large number where dist_points_to_surfaces is nan & 
    # where are_points_inside_box == 0 (points are outside of box)
    # This prevents these invalid points from being chosen.
    cleaned_dist_points_to_surfaces1 = tf.where(tf.is_nan(dist_points_to_surfaces), 
                                                tf.ones_like(dist_points_to_surfaces)*1e8, 
                                                dist_points_to_surfaces)         # (N,6)
    cleaned_dist_points_to_surfaces = tf.where(tf.equal(are_points_inside_box, 0), 
                                               tf.ones_like(cleaned_dist_points_to_surfaces1)*1e8, 
                                               cleaned_dist_points_to_surfaces1) # (N,6)

    # Choose the surface that is closest out of the 6 surfaces
    min_dist_points_to_surfaces = tf.reduce_min(dist_points_to_surfaces, axis=1) # (N,)

  return min_dist_points_to_surfaces

def tf_distance_to_closest_3D_box_surface_multi(point_clouds, box_params_list):

  with tf.variable_scope('distance_to_closest_3D_box_surface_multi'):
    
    d = tf.map_fn(lambda x: [tf_distance_to_closest_3D_box_surface(x[0], (x[1], x[2], x[3]))], 
                  [point_clouds, box_params_list[0], box_params_list[1], box_params_list[2]],
                  dtype=[tf.float32])
    d = tf.concat(d, axis=0) # (B,N)

  return d

def tf_mean_weighted_distance_to_closest_3D_box_surface(point_cloud, mask, box_params, weight_for_points_within=0):
  """
  Calculates mean weighted distance from every point to its closest surfaces on the box.

  Inputs:
    point_cloud: (N,3)
    mask       : (N,)
    box_params : ((3,), (3,), (2,))
  Outputs:
    mean_weighted_dist_points_to_surface: Scalar value for the mean weighted distance from every 
      point to its closest surfaces on the box.
  """
  
  with tf.variable_scope('mean_weighted_distance_to_closest_3D_box_surface'):
    
    min_dist_points_to_surfaces = tf_distance_to_closest_3D_box_surface(point_cloud, box_params) # (N,)

    # Weight points within boxes differently
    points_lie_within = tf_check_if_point_cloud_lie_within_box(point_cloud, box_params)
    min_dist_points_to_surfaces = tf.where(tf.equal(points_lie_within, 1), 
                                           min_dist_points_to_surfaces * weight_for_points_within, 
                                           min_dist_points_to_surfaces)  # (N,)

    # Weight it using the mask & get mean
    weighted_dist_points_to_surface = min_dist_points_to_surfaces * mask # (N,)
    mean_weighted_dist_points_to_surface = tf.reduce_sum(weighted_dist_points_to_surface) # ()
    mean_weighted_dist_points_to_surface = mean_weighted_dist_points_to_surface / (tf.reduce_sum(mask) + 1.)#1e-5) # ()

  return mean_weighted_dist_points_to_surface

def tf_mean_weighted_distance_to_closest_3D_box_surface_multi(point_clouds, masks, box_params_list, weight_for_points_within=0):
  
  with tf.variable_scope('mean_weighted_distance_to_closest_3D_box_surface_multi'):
    
    mwd = tf.map_fn(lambda x: [tf_mean_weighted_distance_to_closest_3D_box_surface(x[0], x[1], (x[2], x[3], x[4]),
                               weight_for_points_within=weight_for_points_within)], 
                    [point_clouds, masks, box_params_list[0], box_params_list[1], box_params_list[2]],
                    dtype=[tf.float32])
    mwd = tf.concat(mwd, axis=0)

  return mwd

def tf_get_box_pc_representation(box_reg, pc):
  """
  Create a representation that is unique to the (box_reg, pc) combination and which will therefore
  represent a box-pc distribution.
  Inputs:
    box_reg: Center (B,3), Dims Reg (B,3), Orient Reg (B,)
    pc     : (B,N,N_channels)
  Output:
    box_pc_rep: (B,N,C)
  """
  with tf.variable_scope('get_box_pc_representation'):

    B, N, _ = pc.get_shape().as_list()
    center, dims_reg, orient_reg = box_reg
    l = dims_reg[0]
    w = dims_reg[1]
    h = dims_reg[2]

    translated_pc_xyz     = pc[:,:,0:3] - tf_expand_tile(center, axis=1, tile=[1,N,1])      # (B,N,3)
    translated_pc_xyz_exp = tf.expand_dims(translated_pc_xyz, axis=3)                       # (B,N,3,1)
    center, surface_pts, surface_norms = tf_create_3D_box_by_surface_centers_multi(box_reg, 
                                         apply_translation=False)                           # (B,3),(B,6,3),(B,6,3)
    surface_norms_exp = tf_expand_tile(surface_norms, axis=1, tile=[1,N,1,1])               # (B,N,6,3)
    ray_from_surface_to_3D_point = tf_expand_tile(translated_pc_xyz, axis=2, tile=[1,1,6,1]) - \
                                   tf_expand_tile(surface_pts, axis=1, tile=[1,N,1,1])      # (B,N,6,3)
    # Perpendicular distance from each plane to the 3D point
    perp_dist_from_surface_to_points = surface_norms_exp * ray_from_surface_to_3D_point        # (B,N,6,3)
    perp_dist_from_surface_to_points = tf.reduce_sum(perp_dist_from_surface_to_points, axis=3) # (B,N,6)

    box_pc_rep = tf.concat([pc, perp_dist_from_surface_to_points], axis=2) # (B,N,N_channels+6)

    return box_pc_rep

# SUNRGBD
def project_upright_depth_to_camera(pc, Rtilt):
  """ project point cloud from depth coord to camera coordinate
      Input: (B,N,3), (B,3,3) Output: (B,N,3)
  """
  # Project upright depth to depth coordinate
  pc2 = tf.matmul(tf.transpose(Rtilt, [0,2,1]), tf.transpose(pc[:,:,0:3], [0,2,1])) # (B,3,N)
  return flip_axis_to_camera(tf.transpose(pc2, [0,2,1]))

# SUNRGBD
def project_upright_depth_to_image(pc, Rtilt, K):
  """ Input: (B,N,3), (B,3,3), (B,3,3) Output: (B,N,2) UV and (B,N,) depth """
  pc2 = project_upright_depth_to_camera(pc, Rtilt)
  uv = tf.matmul(pc2, tf.transpose(K, [0,2,1])) # (B,N,3)
  uv = tf.concat([tf.expand_dims(uv[:,:,0] / uv[:,:,2], axis=2), 
                  tf.expand_dims(uv[:,:,1] / uv[:,:,2], axis=2)], axis=2)
  return uv, pc2[:,:,2]

# SUNRGBD
def flip_axis_to_camera(pc):
  """ Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
      Input and output are both (B,N,3) array
  """
  points = tf.concat([tf.expand_dims(pc[:,:,0], axis=2), 
                      tf.expand_dims(pc[:,:,2], axis=2) * -1, 
                      tf.expand_dims(pc[:,:,1], axis=2)], axis=2) # cam X,Y,Z = depth X,-Z,Y
  return points

# SUNRGBD
def flip_axis_to_depth(pc):
  points = tf.concat([tf.expand_dims(pc[:,:,0], axis=2), 
                      tf.expand_dims(pc[:,:,2], axis=2), 
                      tf.expand_dims(pc[:,:,1], axis=2) * -1], axis=2) # depth X,Y,Z = cam X,-Z,Y
  return points

# SUNRGBD
def project_upright_depth_to_upright_camera(pc):
  return flip_axis_to_camera(pc)

# SUNRGBD
def project_upright_camera_to_upright_depth(pc):
  return flip_axis_to_depth(pc)

# SUNRGBD 
def tf_create_3D_box_by_vertices_multi(box_params, apply_translation=False):
  """ TF layer. Input: box_params ((N,3), (N,3), (N,)), Output: (N,8,3) """

  with tf.variable_scope('create_3D_box_by_vertices_multi'):

    centers, dims_reg, orient_reg = box_params
    N = centers.get_shape()[0].value
    l = tf.slice(dims_reg, [0,0], [-1,1]) # (N,1)
    w = tf.slice(dims_reg, [0,1], [-1,1]) # (N,1)
    h = tf.slice(dims_reg, [0,2], [-1,1]) # (N,1)
    ones = tf.ones([N], dtype=tf.float32)
    zeros = tf.zeros([N], dtype=tf.float32)

    # compute_box_3D (utils.py) (upright depth coords)
    c = tf.cos(-1 * orient_reg)
    s = tf.sin(-1 * orient_reg)
    x_corners = tf.concat([-l/2,l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2], axis=1)
    y_corners = tf.concat([w/2,w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2], axis=1)
    z_corners = tf.concat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], axis=1)
    corners = tf.concat([tf.expand_dims(x_corners,1), tf.expand_dims(y_corners,1), tf.expand_dims(z_corners,1)], axis=1) # (N,3,8)
    row1 = tf.stack([c,-s,zeros], axis=1) # (N,3)
    row2 = tf.stack([s,c,zeros], axis=1)
    row3 = tf.stack([zeros,zeros,ones], axis=1)
    R = tf.concat([tf.expand_dims(row1,1), tf.expand_dims(row2,1), tf.expand_dims(row3,1)], axis=1) # (N,3,3)
    
    corners_3d = tf.matmul(R, corners) # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0,2,1]) # (N,8,3)

    # (YS) Change to upright camera coords
    corners_3d = project_upright_depth_to_upright_camera(corners_3d)
    if apply_translation:
      corners_3d += tf.tile(tf.expand_dims(centers,1), [1,8,1]) # (N,8,3)

    # # get_3d_box (roi_seg_box3d_dataset) (upright camera coords - same as point cloud)
    # c = tf.cos(orient_reg)
    # s = tf.sin(orient_reg)
    # x_corners = tf.concat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], axis=1) # (N,8)
    # y_corners = tf.concat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], axis=1) # (N,8)
    # z_corners = tf.concat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], axis=1) # (N,8)
    # corners = tf.concat([tf.expand_dims(x_corners,1), tf.expand_dims(y_corners,1), tf.expand_dims(z_corners,1)], axis=1) # (N,3,8)
    # row1 = tf.stack([c,zeros,s], axis=1) # (N,3)
    # row2 = tf.stack([zeros,ones,zeros], axis=1)
    # row3 = tf.stack([-s,zeros,c], axis=1)
    # R = tf.concat([tf.expand_dims(row1,1), tf.expand_dims(row2,1), tf.expand_dims(row3,1)], axis=1) # (N,3,3)
    
    # corners_3d = tf.matmul(R, corners) # (N,3,8)
    # if apply_translation:
    #   corners_3d += tf.tile(tf.expand_dims(centers,2), [1,1,8]) # (N,3,8)
    # corners_3d = tf.transpose(corners_3d, perm=[0,2,1]) # (N,8,3)

  return centers, corners_3d

def tf_create_3D_box_by_surface_centers(box_params, apply_translation=False, use_base=True):
  """
  Given a single box_param, return a (6,3) vector representing points on the 6 surfaces of the 3D box 
  whose base is centered at origin, a (6,3) vector representing the inward-facing normalized surface normals
  for each of the surfaces & (3,) translation vector.

  The 6 surfaces are as follows: x front (positive direction), x back, y front, y back, z front, z back.

  Camera coordinate system is used where x is right, y is down, z is front.
  Note: For orientation, it is parameterized by (sin theta, cos theta) instead of just (theta).
  """

  with tf.variable_scope('create_3D_box_by_surface_centers'):

    center, dims_reg, orient_reg = box_params
    l = dims_reg[0]
    w = dims_reg[1]
    h = dims_reg[2]
    sin_theta = tf.sin(orient_reg)
    cos_theta = tf.cos(orient_reg)

    # TODO : Make sure h,w,l are positive
    # assert h > 0 and w > 0 and l > 0

    rotation = tf.convert_to_tensor([[cos_theta, 0., sin_theta],
                                     [0.       , 1., 0.       ],
                                     [-sin_theta,0., cos_theta]])
    surface_pts = tf.convert_to_tensor([[l/2, -l/2, 0.  , 0.  , 0. , 0.  ],  
                                        [0. , 0.  , h/2., -h/2, 0. , 0.  ],
                                        [0. , 0.  , 0.  , 0.  , w/2, -w/2]])

    rotated_surface_pts = tf.transpose(tf.matmul(rotation, surface_pts), [1,0])

    surface_normals = tf.convert_to_tensor([[-1., 1., 0., 0., 0., 0.],  
                                            [0. , 0.,-1., 1., 0., 0.],
                                            [0. , 0., 0., 0.,-1., 1.]])
    # surface_normals = tf.convert_to_tensor([[-l, l , 0., 0., 0., 0.],  
    #                                         [0., 0., -h, h , 0., 0.],
    #                                         [0., 0., 0., 0., -w, w ]])
    # surface_normals = tf.nn.l2_normalize(surface_normals, dim=0)
    rotated_surface_normals = tf.transpose(tf.matmul(rotation, surface_normals), [1,0])

    if apply_translation:
      rotated_surface_pts = rotated_surface_pts + center

    # Center that is returned is placed at the center of the box instead of the base
    if not use_base:
      center += tf.convert_to_tensor([0., 0., 0.])

  return [center, rotated_surface_pts, rotated_surface_normals]

def tf_create_3D_box_by_surface_centers_multi(box_params, apply_translation=False):
  """
  Given a list of box_params (N, 8), create centers (N,3), 3D boxes (N,6,3) and 
  surface normals (N,6,3).
  """

  with tf.variable_scope('create_3D_box_by_surface_centers_multi'):

    centers, surface_pts, surface_normals = tf.map_fn(lambda x: tf_create_3D_box_by_surface_centers((x[0], x[1], x[2]), apply_translation=apply_translation), 
                                                      [box_params[0], box_params[1], box_params[2]], dtype=[tf.float32, tf.float32, tf.float32])

  return centers, surface_pts, surface_normals

def tf_create_3D_box_by_corner_and_axes(box_params):
  """
  Given a single box_param, return a (3',3) vector representing the axes vectors that are pointing towards
  the l, h, w directions of the box where the norm of this vector give l, h, w respectively.
  Also returns a (3,) vector representing the corner (where x, y, z are at minimum) of the box.

  Camera coordinate system is used where x is right (l), y is down (h), z is front (w).
  """

  with tf.variable_scope('create_3D_box_by_corner_and_axes'):

    center, dims_reg, orient_reg = box_params
    l = dims_reg[0]
    w = dims_reg[1]
    h = dims_reg[2]
    sin_theta = tf.sin(orient_reg)
    cos_theta = tf.cos(orient_reg)

    # TODO : Make sure h,w,l are positive
    # assert h > 0 and w > 0 and l > 0

    rotation = tf.convert_to_tensor([[cos_theta, 0., sin_theta],
                                     [0.       , 1., 0.       ],
                                     [-sin_theta,0., cos_theta]])
    axes = tf.convert_to_tensor([[l , 0., 0.],  
                                 [0., h , 0.],
                                 [0., 0., w ]])
    rotated_axes = tf.transpose(tf.matmul(rotation, axes), [1,0])

    corner = tf.convert_to_tensor([[-l/2], [-h/2], [-w/2]])
    rotated_corner = tf.squeeze(tf.matmul(rotation, corner), [1])
    rotated_corner = rotated_corner + center

    return [rotated_corner, rotated_axes]

def tf_create_3D_box_by_corner_and_axes_multi(box_params):
  
  with tf.variable_scope('create_3D_box_by_corner_and_axes_multi'): 
  
    corners, axes = tf.map_fn(lambda x: tf_create_3D_box_by_corner_and_axes((x[0], x[1], x[2])), 
                              [box_params[0], box_params[1], box_params[2]], dtype=[tf.float32, tf.float32])

  return corners, axes 

def tf_convert_box_params_from_anchor_to_reg_format(box_params, y_class, dims_anchors, orient_anchors):
  """
  Convert the box params (in anchor format) to a regression format.
  Inputs:
    box_params    : (center (3,), dims_cls (N_dims,), dims_reg (N_dims,3),
                     orient_cls (N_bins,), orient_reg (N_bins,))
    y_class       : ()
    dims_anchors  : (N_dims,3)
    orient_anchors: (N_bins,)
  """
  N_dims, _ = dims_anchors.get_shape().as_list()
  N_bins    = orient_anchors.get_shape().as_list()

  with tf.variable_scope('tf_convert_box_params_from_anchor_to_reg_format'):
    center, dims_cls, dims_reg, orient_cls, orient_reg = box_params
    
    dim_cls_chosen       = tf.argmax(dims_cls, output_type=tf.int32)
    dim_anchor_chosen    = dims_anchors[dim_cls_chosen,:] # (3,)
    dim_reg_chosen       = dims_reg[dim_cls_chosen,:] # (3,)

    orient_cls_chosen    = tf.argmax(orient_cls, output_type=tf.int32) 
    orient_anchor_chosen = orient_anchors[orient_cls_chosen] 
    orient_reg_chosen    = orient_reg[orient_cls_chosen]

    dims   = dim_anchor_chosen + dim_reg_chosen
    # Exponent to prevent negative values but allow dims_reg to continue to have gradients
    # Switches to exp for (-inf, 0.05] and to linear for (0.05, inf)
    dims   = tf.maximum(dims, 1e-5)
    #dims = tf.where(dims > 0, dims, tf.exp(dims))
    #dims = tf.where(dims > 5e-2, dims, 5e-2 * tf.exp(dims - 5e-2))
    orient = orient_anchor_chosen + orient_reg_chosen

  return [center, dims, orient]

def tf_convert_box_params_from_anchor_to_reg_format_multi(box_params, y_classes, dims_anchors, orient_anchors):

  with tf.variable_scope('tf_convert_box_params_from_anchor_to_reg_format_multi'):
    centers, dims, orients = tf.map_fn(lambda x: tf_convert_box_params_from_anchor_to_reg_format(x[0], x[1], dims_anchors, orient_anchors), 
                                       [box_params, y_classes], dtype=[tf.float32, tf.float32, tf.float32])

  return centers, dims, orients 


# SUNRGBD
def tf_rot_box_params(box_param, angle):

  with tf.variable_scope('tf_rot_box_params'):

    center, dims, orient = box_param
    x, y, z = center[0], center[1], center[2]
    cos_theta = tf.squeeze(tf.cos(angle))
    sin_theta = tf.squeeze(tf.sin(angle))

    # Rotate along y-axis
    new_x = cos_theta * x + sin_theta * z
    new_y = y
    new_z = -sin_theta * x + cos_theta * z
    new_center = tf.convert_to_tensor([new_x, new_y, new_z])

    new_orient = tf.squeeze(orient + angle)

  return [new_center, dims, new_orient]

# SUNRGBD
def tf_rot_box_params_multi(box_params, angles):

  with tf.variable_scope('tf_rot_box_params_multi'):

    rot_box_params = tf.map_fn(lambda x: tf_rot_box_params(x[0], x[1]), 
                               [box_params, angles], 
                               dtype=[tf.float32, tf.float32, tf.float32])
    
  return rot_box_params

def tf_roty(points, angle):
  """
  Rotates points (N,D) about the y-axis by an angle.

  Inputs: (N,D)
  angle:  Scalar angle in radians.
  """
  with tf.variable_scope('roty'):

    points_xyz = points[:,:3]
    points_rem = points[:,3:]

    cos_theta = tf.cos(angle)
    sin_theta = tf.sin(angle)
    points_xyz_T = tf.transpose(points_xyz, [1,0]) # (3,N)
    rotation_mat = tf.convert_to_tensor([[cos_theta, 0., sin_theta],
                                         [0.       , 1., 0.       ],
                                         [-sin_theta,0., cos_theta]]) # (3,3)
    rotated_points = tf.matmul(rotation_mat, points_xyz_T)   # (3,N)
    rotated_points = tf.transpose(rotated_points, [1,0]) # (N,3)
    final_points = tf.concat([rotated_points, points_rem], axis=1)

  return final_points

def tf_roty_diff_angles_multi(points, angles):
  """
  Rotates a batch of points (B,N,4) about the y-axis by different angles.
  NOTE: This rotates all the point clouds by different angles.

  Inputs:
    points: (B,N,4)
    angle : (B,)
  """
  with tf.variable_scope('roty_diff_angles_multi'): 
  
    rotated_points = tf.map_fn(lambda x: [tf_roty(x[0], x[1])], 
                               [points, angles], dtype=[tf.float32])

  return rotated_points[0]

def tf_roty_multi(points, angle):
  """
  Rotates a batch of points (B,N,4) about the y-axis by a specified angle.
  NOTE: This rotates all the point clouds by the same angle.

  Inputs:
    points: (B,N,4)
    angle : Scalar angle in radians.
  """
  B, _, _ = points.get_shape().as_list()

  with tf.variable_scope('roty_multi'):

    cos_theta = tf.cos(angle)
    sin_theta = tf.sin(angle)
    points_T = tf.transpose(points, [0,2,1]) # (B,4,N)
    rotation_mat = tf.convert_to_tensor([[cos_theta, 0., sin_theta, 0.],
                                         [0.       , 1., 0.       , 0.],
                                         [-sin_theta,0., cos_theta, 0.],
                                         [0.       , 0., 0.       , 1.]]) # (4,4)
    rotation_mat_exp = tf_expand_tile(rotation_mat, axis=0, tile=[B,1,1]) # (B,4,4)
    rotated_points = tf.matmul(rotation_mat_exp, points_T) # (B,4,N)
    rotated_points = tf.transpose(rotated_points, [0,2,1]) # (B,N,4)

  return rotated_points

def tf_expand_tile(tensor, axis, tile):
  return tf.tile(tf.expand_dims(tensor, axis=axis), tile)


""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2017
"""

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device("/cpu:0"):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    assert(data_format=='NHWC' or data_format=='NCHW')
    if data_format == 'NHWC':
      num_in_channels = inputs.get_shape()[-1].value
    elif data_format=='NCHW':
      num_in_channels = inputs.get_shape()[1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding,
                           data_format=data_format)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn',
                                      data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      assert(data_format=='NHWC' or data_format=='NCHW')
      if data_format == 'NHWC':
        num_in_channels = inputs.get_shape()[-1].value
      elif data_format=='NCHW':
        num_in_channels = inputs.get_shape()[1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding,
                             data_format=data_format)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn',
                                        data_format=data_format)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=None,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
  """ 2D convolution transpose with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor

  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_output_channels, num_in_channels] # reversed to conv2d
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      
      # from slim.convolution2d_transpose
      def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
          dim_size *= stride_size

          if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
          return dim_size

      # caculate output shape
      batch_size = inputs.get_shape()[0].value
      height = inputs.get_shape()[1].value
      width = inputs.get_shape()[2].value
      out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
      out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
      output_shape = [batch_size, out_height, out_width, num_output_channels]

      outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 3D convolution with non-linear operation.

  Args:
    inputs: 5-D tensor variable BxDxHxWxC
    num_output_channels: int
    kernel_size: a list of 3 ints
    scope: string
    stride: a list of 3 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_d, kernel_h, kernel_w,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.conv3d(inputs, kernel,
                           [1, stride_d, stride_h, stride_w, 1],
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
    
    if bn:
      outputs = batch_norm_for_conv3d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D avg pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs


def batch_norm_template_unused(inputs, is_training, scope, moments_dims, bn_decay):
  """ NOTE: this is older version of the util func. it is deprecated.
  Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = _variable_on_cpu(name='beta',shape=[num_channels],
                            initializer=tf.constant_initializer(0))
    gamma = _variable_on_cpu(name='gamma',shape=[num_channels],
                            initializer=tf.constant_initializer(1.0))
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    # Need to set reuse=False, otherwise if reuse, will see moments_1/mean/ExponentialMovingAverage/ does not exist
    # https://github.com/shekkizh/WassersteinGAN.tensorflow/issues/3
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed

def batch_norm_template(inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
      data_format:   'NHWC' or 'NCHW'
  Return:
      normed:        batch-normalized maps
  """
  bn_decay = bn_decay if bn_decay is not None else 0.9
  return tf.contrib.layers.batch_norm(inputs, 
                                      center=True, scale=True,
                                      is_training=is_training, decay=bn_decay,updates_collections=None,
                                      scope=scope,
                                      data_format=data_format)

def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)

def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, data_format):
  """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay, data_format)

def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, data_format):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay, data_format)

def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)

def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs
