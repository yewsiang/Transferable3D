
""" Weakly/Semi-supervised Loss functions.

Author: Tang Yew Siang
Date: July 2019
"""

import tf_util
import tensorflow as tf

##########################################################################################################
#   Weak Box Losses
########################################################################################################## 

def loss_for_deviation_from_range(val, lower_b, upper_b, loss='huber'):
    """
    Penalize for deviating away from lower_b (-inf, lower_b) or upper_b (upper_b, inf). 
    Not penalized only if it stays within bounds (lower_b, upper_b).
    Inputs:
        val: (B,)
        lower_b: (B,)
        upper_b: (B,)
    """
    #tf.assert_greater(upper_b, lower_b)
    is_violate_lower_b = tf.where(val < lower_b, tf.ones_like(val), tf.zeros_like(val))
    is_violate_upper_b = tf.where(val > upper_b, tf.ones_like(val), tf.zeros_like(val))
    if loss == 'huber':
        lower_b_violation = tf.losses.huber_loss(lower_b, val, reduction=tf.losses.Reduction.NONE)
        upper_b_violation = tf.losses.huber_loss(upper_b, val, reduction=tf.losses.Reduction.NONE)
    elif loss == 'mse':
        lower_b_violation = tf.losses.mean_squared_error(lower_b, val, reduction=tf.losses.Reduction.NONE)
        upper_b_violation = tf.losses.mean_squared_error(upper_b, val, reduction=tf.losses.Reduction.NONE)
    else:
        raise Exception('Not implemented: %s' % loss)
    loss = (is_violate_lower_b * lower_b_violation) + (is_violate_upper_b * upper_b_violation)
    return loss

def get_inactive_volume_loss_v1(dims_reg, y_class, inactive_vol_train_classes, num_classes, inactive_vol_loss_margins, scope=None):
    """
    Inputs:
        dims_reg: (B,3)
        y_class: (B,)
        num_classes: ()
        inactive_vol_loss_margins: (N_classes)
    """
    with tf.variable_scope(scope):

        N_classes = inactive_vol_loss_margins.get_shape().as_list()[0]
        assert(N_classes == num_classes == len(inactive_vol_train_classes))
        
        groups = tf.dynamic_partition(dims_reg, y_class, num_classes)
        group_losses = []
        for i, (train_on_cls, cls_dims_reg) in enumerate(zip(inactive_vol_train_classes, groups)):

            if not train_on_cls: continue

            N = tf.shape(cls_dims_reg)[0]
            cls_vol = tf.reduce_prod(cls_dims_reg, axis=1)                      # (?,)
            cls_vol_margin = inactive_vol_loss_margins[i]
            cls_vol_margin_violation = tf.maximum(0., cls_vol_margin - cls_vol) # (?,)
            violation = tf.reduce_mean(cls_vol_margin_violation)
            violation = tf.where(tf.is_nan(violation), tf.zeros_like(violation), violation)
            group_losses.append(violation)

        inactive_volume_loss = tf.reduce_mean(group_losses)

    return inactive_volume_loss

def get_reprojection_loss(pred_box_reg, box2D, Rtilts, Ks, img_dims, rot_frust, use_softmax_projection, softmax_scale_factor, dilate_factor, clip_lower_b_loss, clip_pred_box, loss_type, train_box, reduce_loss=True, scope=None, end_points=None):

    # Assumes that the projection of the GT 3D box will lie between the an inner 2D box (GT 2D box) and an outer 2D box (factor * GT 2D box).
    with tf.variable_scope(scope):
        # Stop gradient if necessary
        center_reg, dims_reg, orient_reg = pred_box_reg
        center_reg = center_reg if train_box[0] else tf.stop_gradient(center_reg)
        dims_reg   = dims_reg   if train_box[1] else tf.stop_gradient(dims_reg)
        orient_reg = orient_reg if train_box[2] else tf.stop_gradient(orient_reg)
        box = (center_reg, dims_reg, orient_reg)

        # Rotate opposite to frustum normalization
        corrected_box = tf_util.tf_rot_box_params_multi(box, 1 * rot_frust)
        _, pred_box_points = tf_util.tf_create_3D_box_by_vertices_multi(corrected_box, apply_translation=True) 
        
        # Calculate the (left, top, right, bot) differently
        if use_softmax_projection:
            # Project onto image plane and take soft maximum (each point will contribute differently)
            pred_projected_box = tf_util.tf_get_2D_bbox_of_softmax_projection_sunrgbd_multi(pred_box_points, Rtilts, Ks,
                                         softmax_scale_factor=softmax_scale_factor) # (B,4)
        else:
            # Project onto image plane and take the maximum
            pred_projected_box = tf_util.tf_get_2D_bbox_of_projection_sunrgbd_multi(pred_box_points, Rtilts, Ks) # (B,4)

        if clip_pred_box:
            # Clip according to img dims
            pred_projected_box = tf_util.tf_clip_2D_bbox_to_image_dims_multi(pred_projected_box, img_dims)
            # Use GT 2D box as lower bound, and a dilated GT 2D box as upper bound for supervision
            small_box = box2D
            small_box = tf_util.tf_clip_2D_bbox_to_image_dims_multi(small_box, img_dims)
            big_box   = tf_util.tf_dilate_2D_bboxes(box2D, dilate_factor=dilate_factor)
            big_box   = tf_util.tf_clip_2D_bbox_to_image_dims_multi(big_box, img_dims)
            # Lower and upper bound changes according to which side it is for small/big box
            # left corresponds to x_min, top corresponds to y_min. Origin is at top left.
            left_loss  = loss_for_deviation_from_range(pred_projected_box[:,0], big_box[:,0], small_box[:,0], loss=loss_type)
            top_loss   = loss_for_deviation_from_range(pred_projected_box[:,1], big_box[:,1], small_box[:,1], loss=loss_type)
            right_loss = loss_for_deviation_from_range(pred_projected_box[:,2], small_box[:,2], big_box[:,2], loss=loss_type)
            bot_loss   = loss_for_deviation_from_range(pred_projected_box[:,3], small_box[:,3], big_box[:,3], loss=loss_type)
            reprojection_loss = left_loss + top_loss + right_loss + bot_loss
        else:
            # Projection of predicted 3D box will not be clipped.
            # Instead, we clip the loss to 0 if the outer 2D box has been clipped.
            # Use GT 2D box as lower bound, and a dilated GT 2D box as upper bound for supervision
            
            if clip_lower_b_loss:
                # Loss for violating the lower bound will also be clipped if the big box was clipped
                small_box    = box2D                                                                # (B,4)
                small_box    = tf_util.tf_clip_2D_bbox_to_image_dims_multi(small_box, img_dims)
                big_box      = tf_util.tf_dilate_2D_bboxes(box2D, dilate_factor=dilate_factor)      # (B,4)
                big_box_clip = tf_util.tf_clip_2D_bbox_to_image_dims_multi(big_box, img_dims)       # (B,4)
                is_big_box_not_clipped = tf.cast(tf.equal(big_box, big_box_clip), dtype=tf.float32) # (B,4)
                # Lower and upper bound changes according to which side it is for small/big box
                # left corresponds to x_min, top corresponds to y_min. Origin is at top left.
                left_loss  = is_big_box_not_clipped[:,0] * loss_for_deviation_from_range(pred_projected_box[:,0], big_box_clip[:,0], small_box[:,0], loss=loss_type)
                top_loss   = is_big_box_not_clipped[:,1] * loss_for_deviation_from_range(pred_projected_box[:,1], big_box_clip[:,1], small_box[:,1], loss=loss_type)
                right_loss = is_big_box_not_clipped[:,2] * loss_for_deviation_from_range(pred_projected_box[:,2], small_box[:,2], big_box_clip[:,2], loss=loss_type)
                bot_loss   = is_big_box_not_clipped[:,3] * loss_for_deviation_from_range(pred_projected_box[:,3], small_box[:,3], big_box_clip[:,3], loss=loss_type)
                reprojection_loss = left_loss + top_loss + right_loss + bot_loss
                reprojection_loss = tf.minimum(reprojection_loss, 1000.)

            else:
                # We treat inner_box violation and outer_box violation differently
                # For inner_box violation, it will always be enforced.
                # For outer_box violation, it will be set to 0 if big box has been clipped.
                small_box    = box2D                                                                # (B,4)
                small_box    = tf_util.tf_clip_2D_bbox_to_image_dims_multi(small_box, img_dims)
                big_box      = tf_util.tf_dilate_2D_bboxes(box2D, dilate_factor=dilate_factor)      # (B,4)
                big_box_clip = tf_util.tf_clip_2D_bbox_to_image_dims_multi(big_box, img_dims)       # (B,4)
                is_big_box_not_clipped = tf.cast(tf.equal(big_box, big_box_clip), dtype=tf.float32) # (B,4)
                # Lower and upper bound changes according to which side it is for small/big box
                # left corresponds to x_min, top corresponds to y_min. Origin is at top left.

                loss = tf.losses.huber_loss if loss_type == 'huber' else tf.losses.mean_squared_error

                # Note the image coordinates are all calculated with origin at top left corner.
                # A positive x & y direction leads to moving right and bottom.
                # Loss for the left side
                left_inner_b_violation = loss(small_box[:,0]   , pred_projected_box[:,0], reduction=tf.losses.Reduction.NONE) # (B,)
                left_outer_b_violation = loss(big_box_clip[:,0], pred_projected_box[:,0], reduction=tf.losses.Reduction.NONE)
                # No left inner box violation if pred_projected_box[:,0] lies on the left side of the (left side of inner box)
                # No left outer box violation if pred_projected_box[:,0] lies on the right side of the (left side of outer box)
                left_inner_b_violation = tf.where(pred_projected_box[:,0] < small_box[:,0]   , tf.zeros_like(left_inner_b_violation), left_inner_b_violation) # (B,)
                left_outer_b_violation = tf.where(pred_projected_box[:,0] > big_box_clip[:,0], tf.zeros_like(left_outer_b_violation), left_outer_b_violation)
                # Outer box violation is only valid where big box is not clipped
                left_outer_b_violation = left_outer_b_violation * is_big_box_not_clipped[:,0]
                left_loss = left_inner_b_violation + left_outer_b_violation # (B,)

                # Loss for the top side (same as left)
                top_inner_b_violation = loss(small_box[:,1]   , pred_projected_box[:,1], reduction=tf.losses.Reduction.NONE) # (B,)
                top_outer_b_violation = loss(big_box_clip[:,1], pred_projected_box[:,1], reduction=tf.losses.Reduction.NONE)
                # No top inner box violation if pred_projected_box[:,1] lies on the top side of the (top side of inner box)
                # No top outer box violation if pred_projected_box[:,1] lies on the right side of the (top side of outer box)
                top_inner_b_violation = tf.where(pred_projected_box[:,1] < small_box[:,1]   , tf.zeros_like(top_inner_b_violation), top_inner_b_violation) # (B,)
                top_outer_b_violation = tf.where(pred_projected_box[:,1] > big_box_clip[:,1], tf.zeros_like(top_outer_b_violation), top_outer_b_violation)
                # Outer box violation is only valid where big box is not clipped
                top_outer_b_violation = top_outer_b_violation * is_big_box_not_clipped[:,1]
                top_loss = top_inner_b_violation + top_outer_b_violation # (B,)

                # Loss for the right side (slightly different signs because of sides)
                right_inner_b_violation = loss(small_box[:,2]   , pred_projected_box[:,2], reduction=tf.losses.Reduction.NONE) # (B,)
                right_outer_b_violation = loss(big_box_clip[:,2], pred_projected_box[:,2], reduction=tf.losses.Reduction.NONE)
                # No right inner box violation if pred_projected_box[:,2] lies on the right side of the (right side of inner box)
                # No right outer box violation if pred_projected_box[:,2] lies on the right side of the (right side of outer box)
                right_inner_b_violation = tf.where(pred_projected_box[:,2] > small_box[:,2]   , tf.zeros_like(right_inner_b_violation), right_inner_b_violation) # (B,)
                right_outer_b_violation = tf.where(pred_projected_box[:,2] < big_box_clip[:,2], tf.zeros_like(right_outer_b_violation), right_outer_b_violation)
                # Outer box violation is only valid where big box is not clipped
                right_outer_b_violation = right_outer_b_violation * is_big_box_not_clipped[:,2]
                right_loss = right_inner_b_violation + right_outer_b_violation # (B,)
                
                # Loss for the bot side (same as right)
                bot_inner_b_violation = loss(small_box[:,3]   , pred_projected_box[:,3], reduction=tf.losses.Reduction.NONE) # (B,)
                bot_outer_b_violation = loss(big_box_clip[:,3], pred_projected_box[:,3], reduction=tf.losses.Reduction.NONE)
                # No bot inner box violation if pred_projected_box[:,3] lies on the bot side of the (bot side of inner box)
                # No bot outer box violation if pred_projected_box[:,3] lies on the right side of the (bot side of outer box)
                bot_inner_b_violation = tf.where(pred_projected_box[:,3] > small_box[:,3]   , tf.zeros_like(bot_inner_b_violation), bot_inner_b_violation) # (B,)
                bot_outer_b_violation = tf.where(pred_projected_box[:,3] < big_box_clip[:,3], tf.zeros_like(bot_outer_b_violation), bot_outer_b_violation)
                # Outer box violation is only valid where big box is not clipped
                bot_outer_b_violation = bot_outer_b_violation * is_big_box_not_clipped[:,3]
                bot_loss = bot_inner_b_violation + bot_outer_b_violation # (B,)

                reprojection_loss = tf.minimum(left_loss, 1000.) + tf.minimum(top_loss, 1000.) + \
                                    tf.minimum(right_loss, 1000.) + tf.minimum(bot_loss, 1000.)

                # Penalize for every point
                # pred_projected_box = tf_util.tf_get_projected_points_sunrgbd_multi(pred_box_points, Rtilts, Ks) # (B,N,2)
                # B, N, _ = pred_projected_box.get_shape().as_list()
                # small_box    = box2D                                                                            # (B,4)
                # small_box    = tf_util.tf_clip_2D_bbox_to_image_dims_multi(small_box, img_dims)                 # (B,4)
                # big_box      = tf_util.tf_dilate_2D_bboxes(box2D, dilate_factor=dilate_factor)                  # (B,4)
                # big_box_clip = tf_util.tf_clip_2D_bbox_to_image_dims_multi(big_box, img_dims)                   # (B,4)
                # small_box    = tf.reshape(tf_util.tf_expand_tile(small_box, axis=1, tile=[1,N,1]), [B*N,4])     # (B*N,4)
                # big_box      = tf.reshape(tf_util.tf_expand_tile(big_box, axis=1, tile=[1,N,1]), [B*N,4])       # (B*N,4)
                # big_box_clip = tf.reshape(tf_util.tf_expand_tile(big_box_clip, axis=1, tile=[1,N,1]), [B*N,4])  # (B*N,4)
                # is_big_box_not_clipped = tf.cast(tf.equal(big_box, big_box_clip), dtype=tf.float32)             # (B*N,4)

                # # Every point will be penalized for deviation from the 4 ranges in all directions.
                # # The final loss chosen for that point will be for the nearest range.
                # # If the big box is clipped for that direction, it means all the losses from the range in that
                # # direction should be invalid (since we do not know where the actual box might extend to).
                # # In the code below, if big box is clipped, is_big_box_not_clipped will be set to 0. When say left_loss is
                # # multipled by 0, it will be 0 (the minimum). Thereafter, the minimum of left_loss and right_loss will 
                # # be left_loss (since it is at the minimum). This means that there will be no penalty if big box is clipped.
                # pred_projected_box = tf.reshape(pred_projected_box, [B*N,2]) # (B*N,2)
                # left_loss  = is_big_box_not_clipped[:,0] * loss_for_deviation_from_range(pred_projected_box[:,0], big_box_clip[:,0], small_box[:,0], loss=loss_type) # (B*N)
                # top_loss   = is_big_box_not_clipped[:,1] * loss_for_deviation_from_range(pred_projected_box[:,1], big_box_clip[:,1], small_box[:,1], loss=loss_type)
                # right_loss = is_big_box_not_clipped[:,2] * loss_for_deviation_from_range(pred_projected_box[:,0], small_box[:,2], big_box_clip[:,2], loss=loss_type)
                # bot_loss   = is_big_box_not_clipped[:,3] * loss_for_deviation_from_range(pred_projected_box[:,1], small_box[:,3], big_box_clip[:,3], loss=loss_type)
                # horz_loss  = tf.reshape(tf.minimum(left_loss, right_loss), [B,N]) # (B,N)
                # vert_loss  = tf.reshape(tf.minimum(top_loss, bot_loss), [B,N])
                # horz_loss  = tf.minimum(horz_loss, 1000.)
                # vert_loss  = tf.minimum(vert_loss, 1000.)
                # reprojection_loss = tf.reduce_mean(horz_loss + vert_loss, axis=1) # (B)


        if reduce_loss:
            reprojection_loss = tf.reduce_mean(reprojection_loss)

        # End_points only used for visualization + debugging
        if end_points is not None:
            end_points['reproj_pred_box3D_pts']  = pred_box_points
            end_points['reproj_proj_pred_box']   = pred_projected_box
            end_points['reproj_gt_box2D_lowerb'] = small_box
            end_points['reproj_gt_box2D_upperb'] = big_box
            end_points['left_loss']              = left_loss
            end_points['top_loss']               = top_loss
            end_points['right_loss']             = right_loss
            end_points['bot_loss']               = bot_loss
            end_points['reproj_loss']            = reprojection_loss

    return reprojection_loss

def get_surface_loss(pred_box_reg, pc_xyz, soft_mask, margin, scale_dims_factor, weight_for_points_within, train_seg, train_box, reduce_loss=True, scope=None, end_points=None):
    
    with tf.variable_scope(scope):
        # Stop gradient if necessary
        center_reg, dims_reg, orient_reg = pred_box_reg
        mask       = soft_mask  if train_seg    else tf.stop_gradient(soft_mask)
        center_reg = center_reg if train_box[0] else tf.stop_gradient(center_reg)
        dims_reg   = dims_reg   if train_box[1] else tf.stop_gradient(dims_reg)
        orient_reg = orient_reg if train_box[2] else tf.stop_gradient(orient_reg)
        box = (center_reg, dims_reg * scale_dims_factor, orient_reg)

        # surface_loss = tf_util.tf_mean_weighted_distance_to_closest_3D_box_surface_multi(pc_xyz, \
        #                     mask, box, weight_for_points_within=weight_for_points_within)
        min_dist_to_closest_3D_box_surface = tf_util.tf_distance_to_closest_3D_box_surface_multi(pc_xyz, box) # (B,N)
        surface_loss = tf.maximum(0., min_dist_to_closest_3D_box_surface - margin)                             # (B,N)
        surface_loss = surface_loss * soft_mask                                                               # (B,N)
        surface_loss = tf.reduce_mean(surface_loss, axis=1)                                                   # (B,)

        if reduce_loss:
            surface_loss = tf.reduce_mean(surface_loss)

        # End_points only used for visualization + debugging
        if end_points is not None:
            end_points['surface_loss'] = surface_loss

    return surface_loss

def get_intraclass_variance_loss_v1(dims_reg, y_class, intraclsdims_train_classes, num_classes, use_margin_loss, dims_sd_margin, loss_type, scope=None):
    
    with tf.variable_scope(scope):
        groups = tf.dynamic_partition(dims_reg, y_class, num_classes)
        group_losses = []
        for i, (train_on_cls, cls_dims_reg) in enumerate(zip(intraclsdims_train_classes, groups)):

            if not train_on_cls: continue

            # cls_dims_reg has dims (?,3)
            N = tf.shape(cls_dims_reg)[0]
            batch_mean_dims = tf.reduce_mean(cls_dims_reg, axis=0)                            # (3,)
            batch_mean_dims_exp = tf_util.tf_expand_tile(batch_mean_dims, axis=0, tile=[N,1]) # (?,3)
            batch_mean_dims_exp = tf.stop_gradient(batch_mean_dims_exp)
            if loss_type == 'huber':
                intraclass_variance_loss = tf.losses.huber_loss(labels=batch_mean_dims_exp, 
                                                                predictions=cls_dims_reg)
            elif loss_type == 'mse':
                intraclass_variance_loss = tf.losses.mean_squared_error(labels=batch_mean_dims_exp, 
                                                                        predictions=cls_dims_reg)
            group_losses.append(intraclass_variance_loss)

        intraclass_variance_loss = tf.reduce_mean(group_losses)

    return intraclass_variance_loss

##########################################################################################################
#   Semi Box Losses
########################################################################################################## 

def get_D_loss(logits_real, logits_fake, loss_type='SIGMOID', use_soft_noisy_labels_D=False, flip_labels_prob=0., mask_real=None, mask_fake=None, reduce_loss=True, scope=None):
    # D(x) closer to 1 means it is predicting that it is real
    with tf.variable_scope(scope):
        # Discriminator needs to Max E[log D(x)]   + E[log (1 - D(G(z)))]
        #                         or E[log D_real] + E[log (1 - D_fake)]
        #                    or Min -E[log D_real] - E[log (1 - D_fake)]
        B_real = logits_real.get_shape().as_list()[0]
        B_fake = logits_fake.get_shape().as_list()[0]
        # L_real = -tf.log(0.01 + tf.sigmoid(logits_real))
        # L_fake = -tf.log(1.01 - tf.sigmoid(logits_fake))
        
        if loss_type == 'SIGMOID':
            logits_real = tf.squeeze(logits_real, axis=1)
            logits_fake = tf.squeeze(logits_fake, axis=1)
            targets_real = tf.random_uniform([B_real], minval=0.7, maxval=1.2) if use_soft_noisy_labels_D else 1
            targets_fake = tf.random_uniform([B_fake], minval=0.,  maxval=0.3) if use_soft_noisy_labels_D else 0
            targets_real_final = targets_real 
            targets_fake_final = targets_fake
            if flip_labels_prob > 0.:
                targets_real_final = tf.where(tf.random_uniform([B_real], minval=0., maxval=1.) > flip_labels_prob, 
                                              targets_real_final, targets_fake)
                targets_fake_final = tf.where(tf.random_uniform([B_fake], minval=0., maxval=1.) > flip_labels_prob, 
                                              targets_fake_final, targets_real)
            L_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, 
                                                             labels=tf.ones_like(logits_real) * targets_real_final)
            L_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, 
                                                             labels=tf.ones_like(logits_fake) * targets_fake_final)
        elif loss_type == 'SOFTMAX':
            targets_real = tf.random_uniform([B_real], minval=0.7, maxval=1.) if use_soft_noisy_labels_D else tf.ones(B_real)
            targets_fake = tf.random_uniform([B_fake], minval=0.,  maxval=0.3) if use_soft_noisy_labels_D else tf.zeros(B_fake)
            targets_real_final = targets_real 
            targets_fake_final = targets_fake
            if flip_labels_prob > 0.:
                targets_real_final = tf.where(tf.random_uniform([B_real], minval=0., maxval=1.) > flip_labels_prob, 
                                              targets_real_final, targets_fake)
                targets_fake_final = tf.where(tf.random_uniform([B_fake], minval=0., maxval=1.) > flip_labels_prob, 
                                              targets_fake_final, targets_real)
            targets_real_final = tf.stack([1 - targets_real_final, targets_real_final], axis=1)
            targets_fake_final = tf.stack([1 - targets_fake_final, targets_fake_final], axis=1)
            L_real = tf.nn.softmax_cross_entropy_with_logits(logits=logits_real, 
                                                             labels=targets_real_final)
            L_fake = tf.nn.softmax_cross_entropy_with_logits(logits=logits_fake, 
                                                             labels=targets_fake_final)


        if mask_real is not None: 
            L_real = L_real * mask_real
            L_real_mean = tf.reduce_sum(L_real) / (tf.reduce_sum(mask_real) + 1e-3)
        else:
            L_real_mean = tf.reduce_mean(L_real)

        if mask_fake is not None: 
            L_fake = L_fake * mask_fake
            L_fake_mean = tf.reduce_sum(L_fake) / (tf.reduce_sum(mask_fake) + 1e-3)
        else:
            L_fake_mean = tf.reduce_mean(L_fake)

        if reduce_loss:
            # Reduce mean separately for logits_real & logits_fake can have different dims
            D_losses = L_real_mean + L_fake_mean
        else:    
            D_losses = L_real + L_fake

    return D_losses

def get_G_loss(logits_fake, reduce_loss=True, scope=None):
    
    with tf.variable_scope(scope):
        # Generator needs to Min E[log (1 - D(G(z)))]
        #                     or E[log (1 - D_fake)]
        pred_for_fake = tf.sigmoid(logits_fake)
        G_losses = -tf.log(0.01 + pred_for_fake)
        #G_losses = tf.log(1 - pred_for_fake)
        if reduce_loss:
            G_losses = tf.reduce_mean(G_losses)

    return G_losses

def huber_loss(error, delta, reduce_loss=True, scope='huber_loss'):

    with tf.variable_scope(scope) as sc:
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = (abs_error - quadratic)
        losses = 0.5 * quadratic**2 + delta * linear
        if reduce_loss:
            losses = tf.reduce_mean(losses)
        return losses