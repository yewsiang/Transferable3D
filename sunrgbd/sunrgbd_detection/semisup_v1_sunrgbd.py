
""" Semi-supervised model defnitions and losses.

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
import weak_losses
import semisup_models
from model_util import get_box3d_corners_sunrgbd, get_box3d_corners_helper
from roi_seg_box3d_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_CLASS, compute_box3d_iou, class2type, type_mean_size
MEAN_DIMS_ARR = np.zeros((NUM_SIZE_CLUSTER, 3))
for i in range(NUM_SIZE_CLUSTER):
    MEAN_DIMS_ARR[i,:] = type_mean_size[class2type[i]]

# For 2D features
INPUT_IMG_CHANNELS = 3


##############################################################################
# Model definitions
##############################################################################

def placeholder_inputs(batch_size, num_point, num_channel):
    # Inputs
    pc_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_channel))
    bg_pc_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_channel))
    img_pl = tf.placeholder(tf.float32, shape=(batch_size, None, None, INPUT_IMG_CHANNELS))
    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, NUM_CLASS))

    # For strong supervision
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    centers_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    y_orient_cls_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    y_orient_reg_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    y_dims_cls_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    y_dims_reg_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    # For semi/weak supervision
    # KITTI Camera parameters
    R0_rect_pl = tf.placeholder(tf.float32, shape=(batch_size,3,3))
    P_pl = tf.placeholder(tf.float32, shape=(batch_size,3,4))
    # SUNRGBD Camera parameters
    Rtilt_pl = tf.placeholder(tf.float32, shape=(batch_size,3,3))
    K_pl = tf.placeholder(tf.float32, shape=(batch_size,3,3))
    # For Reprojection
    rot_frust_pl = tf.placeholder(tf.float32, shape=(batch_size, 1))
    box2D_pl = tf.placeholder(tf.float32, shape=(batch_size, 4))
    img_dim_pl = tf.placeholder(tf.float32, shape=(batch_size, 2))
    is_data_2D_pl = tf.placeholder(tf.int32, shape=(batch_size,))

    return pc_pl, bg_pc_pl, img_pl, one_hot_vec_pl, labels_pl, centers_pl, y_orient_cls_pl, \
        y_orient_reg_pl, y_dims_cls_pl, y_dims_reg_pl, R0_rect_pl, P_pl, Rtilt_pl, K_pl, \
        rot_frust_pl, box2D_pl, img_dim_pl, is_data_2D_pl

def get_semi_model(pc, bg_pc, img, one_hot_vec, is_training, use_one_hot, oracle_mask=None, norm_box2D=None, bn_decay=None, c=None):
    if c.SEMI_MODEL == 'A':
        return get_semi_model_backbone(pc, bg_pc, img, one_hot_vec, is_training, use_one_hot, 
                                oracle_mask=oracle_mask, norm_box2D=norm_box2D, 
                                bn_decay=bn_decay, c=c)
    elif c.SEMI_MODEL == 'F':
        return get_semi_model_final(pc, bg_pc, img, one_hot_vec, is_training, use_one_hot, 
                                oracle_mask=oracle_mask, norm_box2D=norm_box2D, 
                                bn_decay=bn_decay, c=c)
    else:
        raise Exception('Not implemented SEMI_MODEL: %s' % c.SEMI_MODEL)

def get_semi_model_backbone(pc, bg_pc, img, one_hot_vec, is_training, use_one_hot, oracle_mask=None, norm_box2D=None, bn_decay=None, c=None):
    """ 
    For Semi-supervised Training.

    PC (point cloud) and BG_PC (background point cloud) is BxNx6, 
    img is BxNxIMG_FEAT_CHANNELS, onehotvec is Bx3, output BxNx2.
    """
    end_points = { 'point_cloud'   : pc,
                   'class_one_hot' : one_hot_vec,
                   'class_ids'     : tf.cast(tf.argmax(one_hot_vec, axis=1), dtype=tf.int32),
                   'dims_anchors'  : tf.constant(MEAN_DIMS_ARR, dtype=tf.float32),
                   'orient_anchors': tf.constant(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN), 
                                                 dtype=tf.float32) }
    img_feats = None
    if not use_one_hot: one_hot_vec = None
    if not oracle_mask is None: raise NotImplementedError
    if not c.USE_NORMALIZED_BOX2D_AS_FEATS: norm_box2D = None

    # Instance segmentation network
    logits = semisup_models.v1_inst_seg(pc, img_feats, one_hot_vec, end_points, is_training, 
                                        bn_decay=bn_decay, scope='inst_seg')
    end_points['soft_mask'] = tf.nn.softmax(logits)[:,:,1]
    
    # Subtract points' mean
    mask, mask_xyz_mean, pc_xyz, pc_xyz_stage1 = semisup_models.subtract_points_mean(pc, logits, 
                                                                scope='subtract_points_mean')

    # Regress 1st stage center
    stage1_center = semisup_models.v1_tnet(pc_xyz_stage1, mask, mask_xyz_mean, one_hot_vec, 
        end_points, is_training, norm_box2D=norm_box2D, bn_decay=bn_decay, scope='tnet')

    # Subtract 1st stage center
    pc_xyz_submean = semisup_models.subtract_1st_stage_center(pc_xyz, stage1_center, 
                                    scope='subtract_tnet_center')

    # Regress 3D box (values placed in end_points)
    #
    # Use soft mask
    #
    pred_box = semisup_models.v1_box_est(pc_xyz_submean, stage1_center, mask, one_hot_vec, 
        end_points, is_training, norm_box2D=norm_box2D, bn_decay=bn_decay, c=c, scope='box_est')
    end_points['S_pred_box'] = pred_box

    # Convert anchor format into reg format
    pred_box_reg = tf_util.tf_convert_box_params_from_anchor_to_reg_format_multi(pred_box, 
        end_points['class_ids'], end_points['dims_anchors'], end_points['orient_anchors'])
    end_points['S_pred_box_reg'] = pred_box_reg

    pred = (logits, pred_box)
    return pred, end_points

def get_semi_model_final(pc, bg_pc, img, one_hot_vec, is_training, use_one_hot, oracle_mask=None, norm_box2D=None, bn_decay=None, c=None):
    """ 
    PC (point cloud) is BxNx6, 
    onehotvec is Bx3.
    """
    end_points = { 'point_cloud'   : pc,
                   'class_one_hot' : one_hot_vec,
                   'class_ids'     : tf.cast(tf.argmax(one_hot_vec, axis=1), dtype=tf.int32),
                   'dims_anchors'  : tf.constant(MEAN_DIMS_ARR, dtype=tf.float32),
                   'orient_anchors': tf.constant(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN), 
                                                 dtype=tf.float32) }
    img_feats = None
    # if not use_one_hot: one_hot_vec = None
    if not c.USE_NORMALIZED_BOX2D_AS_FEATS: norm_box2D = None

    # Whether to train the different components or not
    #assert(not c.SEMI_TRAIN_SEG_TRAIN_CLASS_AG_SEG)
    #assert(c.SEMI_TRAIN_BOX_TRAIN_CLASS_AG_TNET and c.SEMI_TRAIN_BOX_TRAIN_CLASS_AG_BOX)
    is_training_seg = is_training # if c.SEMI_TRAIN_SEG_TRAIN_CLASS_AG_SEG else tf.squeeze(tf.zeros(1, dtype=tf.bool))
    is_training_tnet = is_training # if c.SEMI_TRAIN_BOX_TRAIN_CLASS_AG_TNET else tf.squeeze(tf.zeros(1, dtype=tf.bool))
    is_training_box = is_training # if c.SEMI_TRAIN_BOX_TRAIN_CLASS_AG_BOX else tf.squeeze(tf.zeros(1, dtype=tf.bool))

    # =================== Class Agnostic Branch ===================
    with tf.variable_scope('class_agnostic'):
        # Instance segmentation network
        logits = semisup_models.v1_inst_seg(pc, img_feats, None, end_points, is_training_seg, 
                                            bn_decay=bn_decay, scope='inst_seg')
        
        # Subtract points' mean
        if not oracle_mask is None: 
            logits = tf.stack([1 - oracle_mask, oracle_mask], axis=2)
        mask, mask_xyz_mean, pc_xyz, pc_xyz_stage1 = semisup_models.subtract_points_mean(pc, logits, 
                                                                    scope='subtract_points_mean')

        # Regress 1st stage center
        stage1_center = semisup_models.v1_tnet(pc_xyz_stage1, mask, mask_xyz_mean, None, 
            end_points, is_training_tnet, norm_box2D=norm_box2D, bn_decay=bn_decay, scope='tnet')

        # Subtract 1st stage center
        pc_xyz_submean = semisup_models.subtract_1st_stage_center(pc_xyz, stage1_center, 
                                        scope='subtract_tnet_center')

        # Regress 3D box (values placed in end_points)
        W_pred_box = semisup_models.v1_box_est(pc_xyz_submean, stage1_center, mask, None, 
            end_points, is_training_box, norm_box2D=norm_box2D, bn_decay=bn_decay, c=c, scope='box_est')
    

    # =================== Class Dependent Branch ===================
    with tf.variable_scope('class_dependent'):
        
        _, N_classes = one_hot_vec.get_shape().as_list()
        curr_feat    = end_points['feats_lv1']                      # (B,feat_dims)
        if use_one_hot: 
            curr_feat = tf.concat([curr_feat, one_hot_vec], axis=1) # (B,feat_dims + C)
        output_dims  = 3 + NUM_HEADING_BIN*2 + NUM_SIZE_CLUSTER*4

        activation_fn = tf.nn.leaky_relu if c.SEMI_ADV_LEAKY_RELU else tf.nn.relu
        last_layer_fn = tf.nn.tanh if c.SEMI_ADV_TANH_FOR_LAST_LAYER_OF_G else activation_fn
        dropout = c.SEMI_ADV_DROPOUTS_FOR_G
        output = semisup_models.mlps_with_dropout(curr_feat, 
                                layers=[512, 256, output_dims],
                                activation_fns=[activation_fn, last_layer_fn, None], 
                                keep_probs=[dropout, dropout, None],
                                is_training=is_training, 
                                bn=True, bn_decay=bn_decay,
                                c=c, scope='box_refine', reuse=None)               # (B,output_dims)

        F_output = output
        # output  = end_points['box_params']
        # F_output = output + delta_output                                    # (B,output_dims)

        center = tf.slice(F_output, [0,0], [-1,3])
        center = center + stage1_center # Bx3
        end_points['F_center'] = center

        heading_scores = tf.slice(F_output, [0,3], [-1,NUM_HEADING_BIN])
        heading_residuals_normalized = tf.slice(F_output, [0,3+NUM_HEADING_BIN], [-1,NUM_HEADING_BIN])
        end_points['F_heading_scores'] = heading_scores # BxNUM_HEADING_BIN
        end_points['F_heading_residuals_normalized'] = heading_residuals_normalized # BxNUM_HEADING_BIN (should be -1 to 1)
        end_points['F_heading_residuals'] = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN) # BxNUM_HEADING_BIN
        
        batch_size = center.get_shape()[0].value
        size_scores = tf.slice(F_output, [0,3+NUM_HEADING_BIN*2], [-1,NUM_SIZE_CLUSTER]) # BxNUM_SIZE_CLUSTER
        size_residuals_normalized = tf.slice(F_output, [0,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER], [-1,NUM_SIZE_CLUSTER*3])
        size_residuals_normalized = tf.reshape(size_residuals_normalized, [batch_size, NUM_SIZE_CLUSTER, 3]) # BxNUM_SIZE_CLUSTERx3
        end_points['F_size_scores'] = size_scores
        end_points['F_size_residuals_normalized'] = size_residuals_normalized
        end_points['F_size_residuals'] = size_residuals_normalized * tf.expand_dims(tf.constant(MEAN_DIMS_ARR, dtype=tf.float32), 0)

    F_pred_box = (end_points['F_center'], end_points['F_size_scores'], end_points['F_size_residuals'], 
                  end_points['F_heading_scores'], end_points['F_heading_residuals'])

    # Convert anchor format into reg format
    F_pred_box_reg = tf_util.tf_convert_box_params_from_anchor_to_reg_format_multi(F_pred_box, 
        end_points['class_ids'], end_points['dims_anchors'], end_points['orient_anchors'])
    end_points['F_pred_box_reg'] = F_pred_box_reg

    pred = (logits, W_pred_box, F_pred_box)
    return pred, end_points

##############################################################################
# Loss functions
##############################################################################

def get_iou_summary(pred_box, y_box, end_points, name_prefix=''):
    pred_center, pred_dims_cls, pred_dims_reg, pred_orient_cls, pred_orient_reg = pred_box
    y_center, y_dims_cls, y_dims_reg, y_orient_cls, y_orient_reg = y_box
    # Compute IOU 3D
    iou2ds, iou3ds = tf.py_func(compute_box3d_iou, 
      [pred_center, pred_orient_cls, pred_orient_reg, pred_dims_cls, pred_dims_reg, \
       y_center, y_orient_cls, y_orient_reg, y_dims_cls, y_dims_reg], [tf.float32, tf.float32])
    tf.summary.scalar('Box_IOU/%siou_2d' % name_prefix, tf.reduce_mean(iou2ds))
    tf.summary.scalar('Box_IOU/%siou_3d' % name_prefix, tf.reduce_mean(iou3ds))
    end_points[name_prefix + 'iou2ds'] = iou2ds 
    end_points[name_prefix + 'iou3ds'] = iou3ds 

def get_semi_loss(pred, labels, end_points, reduce_loss=True, c=None):
    if c.SEMI_MODEL == 'A':
        return get_semi_loss_backbone(pred, labels, end_points, reduce_loss=reduce_loss, c=c)
    elif c.SEMI_MODEL == 'F':
        return get_semi_loss_final(pred, labels, end_points, reduce_loss=reduce_loss, c=c)
    else:
        raise Exception('Not implemented SEMI_MODEL: %s' % c.SEMI_MODEL)

def get_semi_loss_backbone(pred, labels, end_points, reduce_loss=True, c=None):
    """
    Strong: Use GT Seg Losses for Class A ONLY + GT Box Losses for Class A ONLY.
    Weak: Additionally incorporate Weak losses into the total_loss function.
    """
    pred_seg, pred_box = pred
    S_pred_box_reg     = end_points['S_pred_box_reg']
    pc                 = end_points['point_cloud']
    soft_mask          = end_points['soft_mask']
    center_reg, dims_reg, orient_reg = S_pred_box_reg
    y_seg, y_center, y_orient_cls, y_orient_reg, y_dims_cls, y_dims_reg, \
        R0_rect, P, Rtilt_pl, K_pl, rot_frust, box2D, img_dim, is_data_2D = labels
    y_box = (y_center, y_dims_cls, y_dims_reg, y_orient_cls, y_orient_reg)

    # Weak losses
    reprojection_loss = weak_losses.get_reprojection_loss(S_pred_box_reg,
                                    box2D, Rtilt_pl, K_pl, img_dim, rot_frust, 
                                    use_softmax_projection=c.WEAK_REPROJECTION_USE_SOFTMAX_PROJ,
                                    softmax_scale_factor=c.WEAK_REPROJECTION_SOFTMAX_SCALE,
                                    dilate_factor=c.WEAK_REPROJECTION_DILATE_FACTOR,
                                    clip_lower_b_loss=c.WEAK_REPROJECTION_CLIP_LOWERB_LOSS,
                                    clip_pred_box=c.WEAK_REPROJECTION_CLIP_PRED_BOX,
                                    loss_type=c.WEAK_REPROJECTION_LOSS_TYPE,
                                    train_box=c.WEAK_TRAIN_BOX_W_REPROJECTION,
                                    end_points=end_points, # End_points for debugging
                                    reduce_loss=False, scope='reprojection_loss') # (B,)
    tf.summary.scalar('Weak_Loss/reprojection_loss', tf.reduce_mean(reprojection_loss))

    surface_loss = weak_losses.get_surface_loss(S_pred_box_reg, pc[:,:,0:3], soft_mask,
                   margin=c.WEAK_SURFACE_MARGIN,
                   scale_dims_factor=c.WEAK_SURFACE_LOSS_SCALE_DIMS,
                   weight_for_points_within=c.WEAK_SURFACE_LOSS_WT_FOR_INNER_PTS,
                   train_seg=c.WEAK_TRAIN_SEG_W_SURFACE,
                   train_box=c.WEAK_TRAIN_BOX_W_SURFACE,
                   end_points=end_points, # End_points for debugging
                   reduce_loss=False, scope='surface_loss') # (B,)
    tf.summary.scalar('Weak_Loss/surface_loss', tf.reduce_mean(surface_loss))

    weak_loss_fns = c.WEAK_WEIGHT_REPROJECTION * reprojection_loss + \
                    c.WEAK_WEIGHT_SURFACE * surface_loss    # (B,)
    weak_loss = tf.reduce_mean(weak_loss_fns)
    tf.summary.scalar('Weak_Loss/weak_loss', weak_loss)


    # Strong losses
    label = (y_seg, y_center, y_orient_cls, y_orient_reg, y_dims_cls, y_dims_reg)
    mask_losses, strong_losses = get_strong_loss((pred_seg, pred_box), label, end_points,
                                    reduce_loss=False, c=c) # (B,)
    strong_loss = tf.reduce_mean(strong_losses)
    tf.summary.scalar('Strong_Loss/strong_loss', strong_loss)


    # Mask the loss for data that is supposed to be 2D only
    total_losses = tf.cast(1 - is_data_2D, tf.float32) * (mask_losses + strong_losses) + \
                   tf.cast(is_data_2D, tf.float32) * \
                    (weak_loss_fns * c.SEMI_MULTIPLIER_FOR_WEAK_LOSS)
    # total_losses = strong_losses + \
    #                tf.cast(1 - is_data_2D, tf.float32) * mask_losses
    total_loss = tf.reduce_mean(total_losses)
    
    get_iou_summary(pred_box, y_box, end_points)

    if reduce_loss:
        return total_loss
    else:
        return total_losses

def get_semi_loss_final(pred, labels, end_points, reduce_loss=True, c=None):
    pred_seg, W_pred_box, F_pred_box = pred
    y_seg, y_center, y_orient_cls, y_orient_reg, y_dims_cls, y_dims_reg, \
        R0_rect, P, Rtilt, K, rot_frust, box2D, img_dim, is_data_2D = labels
    y_box = (y_center, y_dims_cls, y_dims_reg, y_orient_cls, y_orient_reg)

    # Strong losses for 3D Classes
    label = (y_seg, y_center, y_orient_cls, y_orient_reg, y_dims_cls, y_dims_reg)
    mask_losses, box_losses = get_strong_loss((pred_seg, F_pred_box), label, end_points,
                                               prefix='F_', reduce_loss=False, c=c) # (B,)
    mask_losses = mask_losses * tf.cast(1 - is_data_2D, tf.float32)
    box_losses  = box_losses * tf.cast(1 - is_data_2D, tf.float32)
    mask_loss   = tf.reduce_sum(mask_losses) / (tf.reduce_sum(tf.cast(1 - is_data_2D, tf.float32)) + 1e-3)
    box_loss    = tf.reduce_sum(box_losses) / (tf.reduce_sum(tf.cast(1 - is_data_2D, tf.float32)) + 1e-3)
    strong_loss = mask_loss + box_loss
    tf.summary.scalar('Strong_Loss/mask_loss', mask_loss)
    tf.summary.scalar('Strong_Loss/box_loss', box_loss)
    tf.summary.scalar('Strong_Loss/strong_loss', strong_loss)


    # Weak loss for 2D Classes
    weak_loss_fns    = tf.constant(0.)
    F_pred_box_reg   = end_points['F_pred_box_reg']
    _, F_dims_reg, _ = F_pred_box_reg
    class_ids        = end_points['class_ids']
    if c.WEAK_WEIGHT_INACTIVE_VOLUME != 0:
        # Box predictions will be penalized if it falls belong this margins
        assert(len(c.WEAK_INACTIVE_VOL_LOSS_MARGINS) == 10)
        inactive_vol_margins = tf.convert_to_tensor(c.WEAK_INACTIVE_VOL_LOSS_MARGINS, dtype=tf.float32)
        inactive_vol_train_classes = end_points['inactive_vol_train_classes']
        inactive_vol_loss = weak_losses.get_inactive_volume_loss_v1(
                                        F_dims_reg, class_ids, inactive_vol_train_classes,
                                        num_classes=10,
                                        inactive_vol_loss_margins=inactive_vol_margins,
                                        scope='inactive_vol_loss')
        tf.summary.scalar('Weak_Loss/inactive_vol_loss', inactive_vol_loss)
        weak_loss_fns += c.WEAK_WEIGHT_INACTIVE_VOLUME * inactive_vol_loss

    if c.WEAK_WEIGHT_INTRACLASSVAR != 0:
        intraclsdims_train_classes = end_points['intraclsdims_train_classes']
        intraclass_variance_loss = weak_losses.get_intraclass_variance_loss_v1(
                                               F_dims_reg, class_ids, intraclsdims_train_classes,
                                               use_margin_loss=c.WEAK_DIMS_USE_MARGIN_LOSS,
                                               dims_sd_margin=c.WEAK_DIMS_SD_MARGIN,
                                               loss_type=c.WEAK_DIMS_LOSS_TYPE,
                                               num_classes=10,
                                               scope='intraclass_variance_loss')
        tf.summary.scalar('Weak_Loss/intracls_dims_var_loss', intraclass_variance_loss)
        weak_loss_fns += c.WEAK_WEIGHT_INTRACLASSVAR * intraclass_variance_loss

    if c.WEAK_WEIGHT_REPROJECTION != 0:
        reprojection_losses = weak_losses.get_reprojection_loss(F_pred_box_reg,
                                          box2D, Rtilt, K, img_dim, rot_frust, 
                                          use_softmax_projection=c.WEAK_REPROJECTION_USE_SOFTMAX_PROJ,
                                          softmax_scale_factor=c.WEAK_REPROJECTION_SOFTMAX_SCALE,
                                          dilate_factor=c.WEAK_REPROJECTION_DILATE_FACTOR,
                                          clip_lower_b_loss=c.WEAK_REPROJECTION_CLIP_LOWERB_LOSS,
                                          clip_pred_box=c.WEAK_REPROJECTION_CLIP_PRED_BOX,
                                          loss_type=c.WEAK_REPROJECTION_LOSS_TYPE,
                                          train_box=c.WEAK_TRAIN_BOX_W_REPROJECTION,
                                          end_points=end_points, # End_points for debugging
                                          reduce_loss=False, scope='reprojection_loss') # (B,)

        if c.WEAK_REPROJECTION_ONLY_ON_2D_CLS:
            reprojection_losses_2D = reprojection_losses * tf.cast(is_data_2D, tf.float32)
            weak_loss_fns += c.WEAK_WEIGHT_REPROJECTION * reprojection_losses_2D
            tf.summary.scalar('Weak_Loss/reprojection_loss_2D', tf.reduce_mean(reprojection_losses_2D))
        else:
            weak_loss_fns += c.WEAK_WEIGHT_REPROJECTION * reprojection_losses
            tf.summary.scalar('Weak_Loss/reprojection_loss', tf.reduce_mean(reprojection_losses))
    weak_loss = tf.reduce_mean(weak_loss_fns)
    tf.summary.scalar('Weak_Loss/weak_loss', weak_loss)


    total_loss = strong_loss + \
                 c.SEMI_MULTIPLIER_FOR_WEAK_LOSS * weak_loss

    if c.SEMI_WEIGHT_BOXPC_FIT_LOSS != 0:
        boxpc_fit_prob = end_points['boxpc_fit_prob']
        boxpc_fit_losses = -tf.log(0.01 + boxpc_fit_prob)
        if c.SEMI_BOXPC_FIT_ONLY_ON_2D_CLS:
            boxpc_fit_losses_2D = boxpc_fit_losses * tf.cast(is_data_2D, tf.float32)
            boxpc_fit_loss_2D = tf.reduce_mean(boxpc_fit_losses_2D)
            total_loss += c.SEMI_WEIGHT_BOXPC_FIT_LOSS * boxpc_fit_loss_2D
            tf.summary.scalar('Semi_Loss/boxpc_fit_loss_2D', boxpc_fit_loss_2D)
        else:
            boxpc_fit_loss = tf.reduce_mean(boxpc_fit_losses)
            total_loss += c.SEMI_WEIGHT_BOXPC_FIT_LOSS * boxpc_fit_loss
            tf.summary.scalar('Semi_Loss/boxpc_fit_loss', tf.reduce_mean(boxpc_fit_loss))

    # Class agnostic
    get_iou_summary(W_pred_box, y_box, end_points, name_prefix='W_') 
    # Class dependent (final) - the one we're interested in
    get_iou_summary(F_pred_box, y_box, end_points, name_prefix='')

    if reduce_loss:
        return total_loss
    else:
        raise Exception('Not implemented')

def get_strong_loss(pred, labels, end_points, prefix='', reg_weight=0.001, reduce_loss=True, c=None):

    pred_seg, pred_box = pred
    pred_center, pred_dims_cls, pred_dims_reg, pred_orient_cls, pred_orient_reg = pred_box
    y_seg, y_center, y_orient_cls, y_orient_reg, y_dims_cls, y_dims_reg = labels
    
    # 1. Segmentation loss
    mask_losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                 logits=pred_seg, labels=y_seg), axis=1)
    mask_loss = tf.reduce_mean(mask_losses)
    tf.summary.scalar('Strong_Loss/3d mask loss', mask_loss)


    # 2. Box Estimation loss
    # Center losses
    center_dist = tf.norm(y_center - end_points[prefix + 'center'], axis=-1)
    center_losses = huber_loss(center_dist, delta=2.0, reduce_loss=False)
    center_loss = tf.reduce_mean(center_losses)
    tf.summary.scalar('Strong_Loss/center loss', center_loss)

    stage1_center_dist = tf.norm(y_center - end_points['stage1_center'], axis=-1)
    stage1_center_losses = huber_loss(stage1_center_dist, delta=1.0, reduce_loss=False)
    stage1_center_loss = tf.reduce_mean(stage1_center_losses)
    tf.summary.scalar('Strong_Loss/stage1 center loss', stage1_center_loss)


    # Heading losses
    heading_class_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                           logits=end_points[prefix + 'heading_scores'], labels=y_orient_cls)
    heading_class_loss = tf.reduce_mean(heading_class_losses)
    tf.summary.scalar('Strong_Loss/heading class loss', heading_class_loss)

    y_orient_cls_onehot = tf.one_hot(y_orient_cls, depth=NUM_HEADING_BIN, 
                                     on_value=1, off_value=0, axis=-1)
    print(y_orient_cls_onehot.shape)
    heading_residual_normalized_label = y_orient_reg / (np.pi/NUM_HEADING_BIN)
    heading_residual_normalized_losses = huber_loss(
        tf.reduce_sum(end_points[prefix + 'heading_residuals_normalized'] * \
        tf.to_float(y_orient_cls_onehot), axis=1) - heading_residual_normalized_label, 
        delta=1.0, reduce_loss=False)
    heading_residual_normalized_loss = tf.reduce_mean(heading_residual_normalized_losses)
    tf.summary.scalar('Strong_Loss/heading residual normalized loss', heading_residual_normalized_loss)


    # Size losses
    size_class_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=end_points[prefix + 'size_scores'], labels=y_dims_cls)
    size_class_loss = tf.reduce_mean(size_class_losses)
    tf.summary.scalar('Strong_Loss/size class loss', size_class_loss)

    y_dims_cls_onehot = tf.one_hot(y_dims_cls, depth=NUM_SIZE_CLUSTER, 
                                   on_value=1, off_value=0, axis=-1)  # (B,NUM_SIZE_CLUSTER)
    y_dims_cls_onehot_tiled = tf.tile(tf.expand_dims(tf.to_float(y_dims_cls_onehot), -1), 
                                      [1,1,3])                        # (B,NUM_SIZE_CLUSTER,3)
    predicted_size_residual_normalized = tf.reduce_sum(
        end_points[prefix + 'size_residuals_normalized'] * y_dims_cls_onehot_tiled, axis=[1]) # (B,3)
    
    tmp3 = tf.expand_dims(tf.constant(MEAN_DIMS_ARR, 
                                      dtype=tf.float32), 0)     # (1,NUM_SIZE_CLUSTER,3)
    mean_size_label = tf.reduce_sum(y_dims_cls_onehot_tiled * tmp3, axis=[1]) # (B,3)
    size_residual_label_normalized = y_dims_reg / mean_size_label
    size_normalized_dist = tf.norm(size_residual_label_normalized - \
        predicted_size_residual_normalized, axis=-1)
    size_residual_normalized_losses = huber_loss(size_normalized_dist, delta=1.0, 
                                                 reduce_loss=False)
    size_residual_normalized_loss = tf.reduce_mean(size_residual_normalized_losses)
    tf.summary.scalar('Strong_Loss/size residual normalized loss', size_residual_normalized_loss)
    

    # Corner loss
    # Compute BOX3D corners
    corners_3d = get_box3d_corners_sunrgbd(end_points[prefix + 'center'], end_points[prefix + 'heading_residuals'], 
                                   end_points[prefix + 'size_residuals']) #   (B,NH,NS,8,3)
    gt_mask = tf.tile(tf.expand_dims(y_orient_cls_onehot, 2), [1,1,NUM_SIZE_CLUSTER]) * \
              tf.tile(tf.expand_dims(y_dims_cls_onehot,1), [1,NUM_HEADING_BIN,1]) # (B,NH,NS)
    corners_3d_pred = tf.reduce_sum(tf.to_float(
        tf.expand_dims(tf.expand_dims(gt_mask,-1),-1)) * corners_3d, axis=[1,2])  # (B,8,3)

    heading_bin_centers = tf.constant(np.arange(0, 2*np.pi, 2*np.pi/NUM_HEADING_BIN), 
                                      dtype=tf.float32)                           # (NH,)
    heading_label = tf.expand_dims(y_orient_reg, 1) + \
                    tf.expand_dims(heading_bin_centers, 0)                        # (B,NH)
    heading_label = tf.reduce_sum(tf.to_float(y_orient_cls_onehot) * heading_label, 1)
    mean_sizes = tf.expand_dims(tf.constant(MEAN_DIMS_ARR, dtype=tf.float32), 0)  # (1,NS,3)
    size_label = mean_sizes + tf.expand_dims(y_dims_reg, 1) # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = tf.reduce_sum(tf.expand_dims(tf.to_float(y_dims_cls_onehot), -1) * \
                               size_label, axis=[1])                               # (B,3)
    corners_3d_gt = get_box3d_corners_helper(y_center, heading_label, size_label)  # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper(y_center, heading_label + np.pi,
                                                  size_label)                      # (B,8,3)

    corners_dist = tf.minimum(tf.norm(corners_3d_pred - corners_3d_gt, axis=-1), 
                              tf.norm(corners_3d_pred - corners_3d_gt_flip, axis=-1))
    print(str(corners_dist.shape) + ' (Corners dist)')
    corners_losses = tf.reduce_mean(huber_loss(corners_dist, delta=1.0, reduce_loss=False), 
                                    axis=1)
    corners_loss = tf.reduce_mean(corners_losses)
    tf.summary.scalar('Strong_Loss/corners loss', corners_loss)

    # if reduce_loss:
    #     total_losses = c.STRONG_WEIGHT_CROSS_ENTROPY * mask_loss + \
    #                    c.STRONG_BOX_MULTIPLER * \
    #                    (c.STRONG_WEIGHT_CENTER * center_loss + \
    #                     c.STRONG_WEIGHT_ORIENT_CLS * heading_class_loss + \
    #                     c.STRONG_WEIGHT_DIMS_CLS * size_class_loss + \
    #                     c.STRONG_WEIGHT_ORIENT_REG * heading_residual_normalized_loss + \
    #                     c.STRONG_WEIGHT_DIMS_REG * size_residual_normalized_loss + \
    #                     c.STRONG_WEIGHT_TNET_CENTER * stage1_center_loss) + \
    #                    c.STRONG_WEIGHT_CORNER * corners_loss
    # else:
    #     total_losses = c.STRONG_WEIGHT_CROSS_ENTROPY * mask_losses + \
    #                    c.STRONG_BOX_MULTIPLER * \
    #                    (c.STRONG_WEIGHT_CENTER * center_losses + \
    #                     c.STRONG_WEIGHT_ORIENT_CLS * heading_class_losses + \
    #                     c.STRONG_WEIGHT_DIMS_CLS * size_class_losses + \
    #                     c.STRONG_WEIGHT_ORIENT_REG * heading_residual_normalized_losses + \
    #                     c.STRONG_WEIGHT_DIMS_REG * size_residual_normalized_losses + \
    #                     c.STRONG_WEIGHT_TNET_CENTER * stage1_center_losses) + \
    #                    c.STRONG_WEIGHT_CORNER * corners_losses

    mask_losses = c.STRONG_WEIGHT_CROSS_ENTROPY * mask_losses
    total_losses = c.STRONG_BOX_MULTIPLER * \
                   (c.STRONG_WEIGHT_CENTER * center_losses + \
                    c.STRONG_WEIGHT_ORIENT_CLS * heading_class_losses + \
                    c.STRONG_WEIGHT_DIMS_CLS * size_class_losses + \
                    c.STRONG_WEIGHT_ORIENT_REG * heading_residual_normalized_losses + \
                    c.STRONG_WEIGHT_DIMS_REG * size_residual_normalized_losses + \
                    c.STRONG_WEIGHT_TNET_CENTER * stage1_center_losses) + \
                    c.STRONG_WEIGHT_CORNER * corners_losses

    return mask_losses, total_losses

def huber_loss(error, delta, reduce_loss=True, scope='huber_loss'):

    with tf.variable_scope(scope) as sc:
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = (abs_error - quadratic)
        losses = 0.5 * quadratic**2 + delta * linear
        if reduce_loss:
            losses = tf.reduce_mean(losses)
        return losses

def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor

def convert_raw_y_box_to_reg_format(y_box, one_hot_vec):
    """
    Convert y_box into anchor format and into reg format.
    Note: y_box is raw format and does not have same dimensions as pred_box.
    """
    y_centers, y_orient_cls, y_orient_reg, y_dims_cls, y_dims_reg = y_box
    class_ids = tf.cast(tf.argmax(one_hot_vec, axis=1), dtype=tf.int32)
    dims_anchors = tf.constant(MEAN_DIMS_ARR, dtype=tf.float32)
    orient_anchors = tf.constant(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN), 
                                           dtype=tf.float32)

    # Dims
    dims_cls = tf.one_hot(y_dims_cls, depth=NUM_SIZE_CLUSTER, 
                                   on_value=1, off_value=0, axis=-1)  # (B,NUM_SIZE_CLUSTER)
    dims_reg = tf_util.tf_expand_tile(y_dims_reg, axis=1, tile=[1,NUM_SIZE_CLUSTER,1]) # (B,NUM_SIZE_CLUSTER,3)

    # Orient
    orient_cls = tf.one_hot(y_orient_cls, depth=NUM_HEADING_BIN, 
                            on_value=1, off_value=0, axis=-1)                              # (B,NUM_HEADING_BIN)
    orient_reg = tf_util.tf_expand_tile(y_orient_reg, axis=1, tile=[1,NUM_HEADING_BIN]) # (B,NUM_HEADING_BIN)

    box = (y_centers, dims_cls, dims_reg, orient_cls, orient_reg)
    box_reg = tf_util.tf_convert_box_params_from_anchor_to_reg_format_multi(box, 
                      class_ids, dims_anchors, orient_anchors)
    return box_reg