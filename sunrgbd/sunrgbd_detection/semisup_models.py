
""" Model utility functions.

Author: Tang Yew Siang
Date: July 2019
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FPN_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(FPN_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../models'))

import numpy as np
import tensorflow as tf

import tf_util
from roi_seg_box3d_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, class2type, type_mean_size
MEAN_DIMS_ARR = np.zeros((NUM_SIZE_CLUSTER, 3))
for i in range(NUM_SIZE_CLUSTER):
    MEAN_DIMS_ARR[i,:] = type_mean_size[class2type[i]]

##############################################################################
# General
##############################################################################

def mlps(input_feat, layers, is_training, bn=True, bn_decay=None, c=None, scope=None, reuse=None):
    print('\n  -------- [%s] --------' % scope)
    with tf.variable_scope(scope, reuse=reuse) as sc: 
        net = input_feat
        for i, layer_size in enumerate(layers):
            if len(layers) - 1 == i: # Last layer
                net = tf_util.fully_connected(net, layer_size, scope='fc%d' % i, bn=False, # TODO: Should not do bn here
                                              activation_fn=None, is_training=is_training, 
                                              bn_decay=bn_decay)
            else:
                net = tf_util.fully_connected(net, layer_size, scope='fc%d' % i, bn=bn, 
                                              is_training=is_training, bn_decay=bn_decay)
    return net

def mlps_with_dropout(input_feat, layers, activation_fns, keep_probs, is_training, bn=True, bn_decay=None, c=None, scope=None, reuse=None):
    assert(len(layers) == len(activation_fns) == len(keep_probs))
    print('\n  -------- [%s] --------' % scope)
    with tf.variable_scope(scope, reuse=reuse) as sc: 
        net = input_feat
        for i, layer_size in enumerate(layers):
            if len(layers) - 1 == i: # Last layer
                net = tf_util.fully_connected(net, layer_size, scope='fc%d' % i, 
                                              activation_fn=activation_fns[i],
                                              is_training=is_training, 
                                              bn=False, bn_decay=bn_decay)
            else:
                net = tf_util.fully_connected(net, layer_size, scope='fc%d' % i,  
                                              activation_fn=activation_fns[i],
                                              is_training=is_training, 
                                              bn=bn, bn_decay=bn_decay)
                net = tf_util.dropout(net, keep_prob=keep_probs[i], 
                                      is_training=is_training,
                                      scope='dp%d' % i)
    return net

##############################################################################
# Instance Segmentation
##############################################################################

def v1_inst_seg(point_cloud, img_feats, one_hot_vec, end_points, is_training, bn_decay=None, scope=None):
    print('\n  -------- [%s] --------' % scope)
    with tf.variable_scope(scope) as sc:
        batch_size, num_point, D = point_cloud.get_shape().as_list()
        pc_image = tf.expand_dims(point_cloud, -1)

        # MLPs over point cloud
        net = tf_util.conv2d(pc_image, 64, [1,D],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        point_feat = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(point_feat, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)
        global_feat = tf_util.max_pool2d(net, [num_point,1],
                                         padding='VALID', scope='maxpool')
        print(global_feat.shape)

        # Add class info + image features into global feature vector
        if one_hot_vec is not None:
            global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
        # global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(img_feats, 1), 1)], axis=3)
        # global_feat = tf.concat([global_feat, 
        #                          tf.expand_dims(tf.expand_dims(img_feats, 1), 1), 
        #                          tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
        global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
        concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])
        print(str(global_feat.shape) + ' (Global Feat)')
        print(point_feat.shape)
        print(global_feat_expand.shape)
        print(concat_feat.shape)

        # MLPs over point cloud with global information
        net = tf_util.conv2d(concat_feat, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv6', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv7', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv8', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv9', bn_decay=bn_decay)
        net = tf_util.dropout(net, is_training, 'dp1', keep_prob=0.5)

        logits = tf_util.conv2d(net, 2, [1,1],
                             padding='VALID', stride=[1,1], activation_fn=None,
                             scope='conv10')
        logits = tf.squeeze(logits, [2]) # BxNxC
        print(str(logits.shape) + ' (Logits / Point-wise mask)')

    return logits

##############################################################################
# Utilities
##############################################################################

def subtract_points_mean(point_cloud, logits, scope=None):
    print('\n  -------- [%s] --------' % scope)
    with tf.variable_scope(scope) as sc:
        num_point = point_cloud.get_shape()[1].value
        #net = tf.concat(axis=3, values=[net, tf.expand_dims(tf.slice(point_cloud, [0,0,0], [-1,-1,3]), 2)])
        mask = tf.slice(logits,[0,0,0],[-1,-1,1]) < tf.slice(logits,[0,0,1],[-1,-1,1])
        mask = tf.to_float(mask) # BxNx1
        mask_count = tf.tile(tf.reduce_sum(mask,axis=1,keep_dims=True), [1,1,3]) # Bx1x3
        print(mask.shape)
        point_cloud_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3]) # BxNx3

        # Subtract points mean
        mask_xyz_mean = tf.reduce_sum(tf.tile(mask, [1,1,3])*point_cloud_xyz, axis=1, keep_dims=True) # Bx1x3
        mask_xyz_mean = mask_xyz_mean/tf.maximum(mask_count,1) # Bx1x3
        point_cloud_xyz_stage1 = point_cloud_xyz - tf.tile(mask_xyz_mean, [1,num_point,1])
        print(str(point_cloud_xyz_stage1.shape) + ' (Point cloud xyz stage1)')

        return mask, mask_xyz_mean, point_cloud_xyz, point_cloud_xyz_stage1

def v1_tnet(point_cloud_xyz_stage1, mask, mask_xyz_mean, one_hot_vec, end_points, is_training, norm_box2D=None, bn_decay=None, scope=None):
    print('\n  -------- [%s] --------' % scope)
    with tf.variable_scope(scope) as sc:
        num_point = point_cloud_xyz_stage1.get_shape()[1].value
        # Regress 1st stage center
        net = tf.expand_dims(point_cloud_xyz_stage1, 2)
        print(net.shape)

        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg1-stage1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg2-stage1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg3-stage1', bn_decay=bn_decay)
        mask_expand = tf.tile(tf.expand_dims(mask,-1), [1,1,1,256])
        masked_net = net*mask_expand
        print(masked_net.shape)

        net = tf_util.max_pool2d(masked_net, [num_point,1], padding='VALID', scope='maxpool-stage1')
        net = tf.squeeze(net, axis=[1,2])
        print(net.shape)

        if one_hot_vec is not None:
            net = tf.concat([net, one_hot_vec], axis=1)
        if norm_box2D is not None:
            net = tf.concat([net, norm_box2D], axis=1)
        net = tf_util.fully_connected(net, 256, scope='fc1-stage1', bn=True, is_training=is_training, bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 128, scope='fc2-stage1', bn=True, is_training=is_training, bn_decay=bn_decay)
        stage1_center = tf_util.fully_connected(net, 3, activation_fn=None, scope='fc3-stage1')
        stage1_center = stage1_center + tf.squeeze(mask_xyz_mean, axis=1) # Bx3
        end_points['stage1_center'] = stage1_center

        return stage1_center

def subtract_1st_stage_center(point_cloud_xyz, stage1_center, scope=None):
    print('\n  -------- [%s] --------' % scope)
    with tf.variable_scope(scope) as sc:
        point_cloud_xyz_submean = point_cloud_xyz - tf.expand_dims(stage1_center, 1)
        print(str(point_cloud_xyz_submean.shape) + ' (Point cloud xyz submean)')
        return point_cloud_xyz_submean

##############################################################################
# Box Estimation
##############################################################################

def v1_box_est(point_cloud_xyz_submean, stage1_center, mask, one_hot_vec, end_points, is_training, norm_box2D=None, bn_decay=None, prefix='', c=None, scope=None):
    print('\n  -------- [%s] --------' % scope)
    with tf.variable_scope(scope) as sc:
        batch_size = point_cloud_xyz_submean.get_shape()[0].value
        num_point = point_cloud_xyz_submean.get_shape()[1].value

        net = tf.expand_dims(point_cloud_xyz_submean, 2)
        print(net.shape)

        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg4', bn_decay=bn_decay)
        mask_expand = tf.tile(tf.expand_dims(mask,-1), [1,1,1,512])
        masked_net = net*mask_expand
        print(masked_net.shape)

        net = tf_util.max_pool2d(masked_net, [num_point,1], padding='VALID', scope='maxpool2')
        net = tf.squeeze(net, axis=[1,2])
        end_points[prefix + 'feats_lv1'] = net
        print(net.shape)

        if one_hot_vec is not None:
            net = tf.concat([net, one_hot_vec], axis=1)
        if norm_box2D is not None:
            net = tf.concat([net, norm_box2D], axis=1)
        net = tf_util.fully_connected(net, 512, scope='fc1', bn=True, is_training=is_training, bn_decay=bn_decay)
        end_points[prefix + 'feats_lv2'] = net
        net = tf_util.fully_connected(net, 256, scope='fc2', bn=True, is_training=is_training, bn_decay=bn_decay)
        end_points[prefix + 'feats_lv3'] = net

        # Predictions
        # First 3 are cx,cy,cz, next NUM_HEADING_BIN*2 are for heading
        # next NUM_SIZE_CLUSTER*4 are for dimension
        output = tf_util.fully_connected(net, 3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')
        end_points[prefix + 'box_params'] = output
        print(str(output.shape) + ' (3 + 2 * Heading(%d) + 4 * Sizes(%d))' % (NUM_HEADING_BIN, NUM_SIZE_CLUSTER))

        center = tf.slice(output, [0,0], [-1,3])
        center = center + stage1_center # Bx3
        end_points[prefix + 'center'] = center
        print('Center                 : ' + str(center.shape))

        heading_scores = tf.slice(output, [0,3], [-1,NUM_HEADING_BIN])
        heading_residuals_normalized = tf.slice(output, [0,3+NUM_HEADING_BIN], [-1,NUM_HEADING_BIN])
        end_points[prefix + 'heading_scores'] = heading_scores # BxNUM_HEADING_BIN
        end_points[prefix + 'heading_residuals_normalized'] = heading_residuals_normalized # BxNUM_HEADING_BIN (should be -1 to 1)
        end_points[prefix + 'heading_residuals'] = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN) # BxNUM_HEADING_BIN
        print('Heading Scores         : ' + str(end_points[prefix + 'heading_scores'].shape))
        print('Heading Residuals Norm : ' + str(end_points[prefix + 'heading_residuals_normalized'].shape))
        print('Heading Residuals      : ' + str(end_points[prefix + 'heading_residuals'].shape))
        
        size_scores = tf.slice(output, [0,3+NUM_HEADING_BIN*2], [-1,NUM_SIZE_CLUSTER]) # BxNUM_SIZE_CLUSTER
        size_residuals_normalized = tf.slice(output, [0,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER], [-1,NUM_SIZE_CLUSTER*3])
        size_residuals_normalized = tf.reshape(size_residuals_normalized, [batch_size, NUM_SIZE_CLUSTER, 3]) # BxNUM_SIZE_CLUSTERx3
        end_points[prefix + 'size_scores'] = size_scores
        end_points[prefix + 'size_residuals_normalized'] = size_residuals_normalized
        end_points[prefix + 'size_residuals'] = size_residuals_normalized * tf.expand_dims(tf.constant(MEAN_DIMS_ARR, dtype=tf.float32), 0)
        print('Size Scores            : ' + str(end_points[prefix + 'size_scores'].shape))
        print('Size Residuals Norm    : ' + str(end_points[prefix + 'size_residuals_normalized'].shape))
        print('Size Residuals         : ' + str(end_points[prefix + 'size_residuals'].shape))

        pred_box = (center, end_points[prefix + 'size_scores'], end_points[prefix + 'size_residuals'], \
                    end_points[prefix + 'heading_scores'], end_points[prefix + 'heading_residuals'])
    return pred_box

##############################################################################
# Box-PC Models
##############################################################################

def box_pc_mask_features_model(box, pc, logits, num_outputs, is_training, end_points, reuse, bn_for_output, 
    normalize_pc=False, normalize_method='SD', one_hot_vec=None, norm_box2D=None, bn_decay=None, c=None, scope=None):

    if c.BOX_PC_MASK_REPRESENTATION == 'A':
        # Combined BoxPC representation
        # box is represented in reg format - (center, dims_reg, orient_reg)
        logits, feats = combined_box_pc_mask_features_model(box, pc, logits, num_outputs, is_training, 
                        end_points=end_points, reuse=reuse, 
                        normalize_pc=normalize_pc, 
                        normalize_method=normalize_method,
                        bn_for_output=False,
                        one_hot_vec=one_hot_vec,
                        norm_box2D=None,
                        bn_decay=bn_decay, c=c, scope='box_pc_mask_model')
    elif c.BOX_PC_MASK_REPRESENTATION == 'B':
        # Independent BoxPC representation
        # box is represented in output space - (center, dims_cls, dims_reg, orient_cls, orient_reg)
        logits, feats = independent_box_pc_mask_features_model(box, pc, logits, num_outputs, is_training, 
                        end_points=end_points, reuse=reuse, 
                        normalize_pc=normalize_pc, 
                        normalize_method=normalize_method,
                        bn_for_output=False,
                        one_hot_vec=one_hot_vec,
                        norm_box2D=None,
                        bn_decay=bn_decay, c=c, scope='box_pc_mask_model')
    else:
        raise Exception('Box pc mask representation not implemented: %s' % c.BOX_PC_MASK_REPRESENTATION)
    return logits, feats

def combined_box_pc_mask_features_model(box_reg, pc, mask, num_outputs, is_training, end_points, reuse, bn_for_output, 
    normalize_pc=False, normalize_method='SD', one_hot_vec=None, norm_box2D=None, bn_decay=None, c=None, scope=None):

    print('\n  -------- [%s] --------' % scope)
    with tf.variable_scope(scope, reuse=reuse):
        batch_size = pc.get_shape()[0].value
        num_point = pc.get_shape()[1].value

        box_pc_rep = tf_util.tf_get_box_pc_representation(box_reg, pc) # (B,N,C+6)
        if normalize_pc:
            box_pc_rep_xyz  = box_pc_rep[:,:,:3]
            box_pc_rep_rest = box_pc_rep[:,:,3:]
            if normalize_method == 'SD':
                norm_box_pc_rep_xyz = tf_util.tf_normalize_point_clouds_to_mean_zero_and_unit_var(box_pc_rep_xyz)
            elif normalize_method == 'Spread':
                norm_box_pc_rep_xyz = tf_util.tf_normalize_point_clouds_to_01(box_pc_rep_xyz)
            else:
                raise Exception('Invalid normalization method')
            box_pc_rep = tf.concat([norm_box_pc_rep_xyz, box_pc_rep_rest], axis=2) # (B,N,C+6)

        if mask is not None:
            box_pc_mask_rep = tf.concat([box_pc_rep, mask], axis=2) # (B,N,C+6+1)
            box_pc_mask_rep = tf.expand_dims(box_pc_mask_rep, -1)   # (B,N,C+6+1,1)
        else:
            box_pc_mask_rep = tf.expand_dims(box_pc_rep, -1)        # (B,N,C+6,1)
        D = box_pc_mask_rep.get_shape()[2].value

        # CONV
        net = tf_util.conv2d(box_pc_mask_rep, 128, [1,D],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg4', bn_decay=bn_decay)

        if mask is not None:
            mask_expand = tf.tile(tf.expand_dims(mask,-1), [1,1,1,512])
            net  = net * mask_expand

        net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='maxpool2')
        net = tf.squeeze(net, axis=[1,2])

        # FC
        if one_hot_vec is not None:
            net = tf.concat([net, one_hot_vec], axis=1)
        if norm_box2D is not None:
            net = tf.concat([net, norm_box2D], axis=1)
        net = tf.reshape(net, [batch_size, -1])
        features_lv1 = net
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        features_lv2 = net
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        features_lv3 = net
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
        net = tf_util.fully_connected(net, num_outputs, bn=bn_for_output, is_training=is_training, 
                                      activation_fn=None, scope='fc3', bn_decay=bn_decay)

        features = { '%s_feats_lv1' % scope : features_lv1,
                     '%s_feats_lv2' % scope : features_lv2,
                     '%s_feats_lv3' % scope : features_lv3 }

    return net, features

def independent_box_pc_mask_features_model(box_reg, pc, mask, num_outputs, is_training, end_points, reuse, bn_for_output, 
    normalize_pc=False, normalize_method='SD', one_hot_vec=None, norm_box2D=None, bn_decay=None, c=None, scope=None):

    print('\n  -------- [%s] --------' % scope)
    with tf.variable_scope(scope, reuse=reuse):
        batch_size, num_point, D = pc.get_shape().as_list()

        # Extract features from box
        box_reg = tf.concat([box_reg[0], box_reg[1], tf.expand_dims(box_reg[2], axis=1)], axis=1) # (B,7) from ((B,3), (B,3), (B,))
        box_feat = mlps(box_reg, [128, 128, 256, 512], is_training, 
                        bn=True, bn_decay=bn_decay, c=c, scope='extract_box_feats', reuse=reuse)  # (B,512)

        # Extract features from pc
        if normalize_pc:
            pc_xyz  = pc[:,:,:3]
            pc_rest = pc[:,:,3:]
            if normalize_method == 'SD':
                norm_pc_xyz = tf_util.tf_normalize_point_clouds_to_mean_zero_and_unit_var(pc_xyz)
            elif normalize_method == 'Spread':
                norm_pc_xyz = tf_util.tf_normalize_point_clouds_to_01(pc_xyz)
            else:
                raise Exception('Invalid normalization method')
            pc = tf.concat([norm_pc_xyz, pc_rest], axis=2)
        
        input_image = tf.expand_dims(pc, -1)
        net = tf_util.conv2d(input_image, 128, [1,D],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 512, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv-reg4', bn_decay=bn_decay)
        if mask is not None:
            mask_expand = tf.tile(tf.expand_dims(mask,-1), [1,1,1,512])
            net  = net * mask_expand
        net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='maxpool2')
        net = tf.squeeze(net, axis=[1,2]) # (B,512)

        # Combine features
        net = tf.concat([box_feat, net], axis=1) # (B,1024)
        features_lv1 = net

        # FC
        if norm_box2D is not None:
            net = tf.concat([net, norm_box2D], axis=1)
        if one_hot_vec is not None:
            net = tf.concat([net, one_hot_vec], axis=1)
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        features_lv2 = net
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
        features_lv3 = net
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp3')
        net = tf_util.fully_connected(net, num_outputs, bn=bn_for_output, is_training=is_training, 
                                      activation_fn=None, scope='fc4', bn_decay=bn_decay)

        features = { '%s_feats_lv1' % scope : features_lv1,
                     '%s_feats_lv2' % scope : features_lv2,
                     '%s_feats_lv3' % scope : features_lv3 }

    return net, features