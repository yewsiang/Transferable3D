
""" BoxPC model defnitions and losses.

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
from roi_seg_box3d_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_CLASS, class2type, type_mean_size
MEAN_DIMS_ARR = np.zeros((NUM_SIZE_CLUSTER, 3))
for i in range(NUM_SIZE_CLUSTER):
    MEAN_DIMS_ARR[i,:] = type_mean_size[class2type[i]]


##############################################################################
# Model definitions
##############################################################################

def placeholder_inputs(batch_size, num_point, num_channels):
    # Inputs
    pc_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_channels))
    one_hot_vec_pl = tf.placeholder(tf.float32, shape=(batch_size, NUM_CLASS))

    # For strong supervision
    y_seg_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    x_center_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    x_orient_cls_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    x_orient_reg_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    x_dims_cls_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    x_dims_reg_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    # Box PC
    y_box_iou_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    y_center_delta_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
    y_orient_delta_pl = tf.placeholder(tf.float32, shape=(batch_size,))
    y_dims_delta_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))

    return pc_pl, one_hot_vec_pl, y_seg_pl, x_center_pl, x_orient_cls_pl, x_orient_reg_pl, \
        x_dims_cls_pl, x_dims_reg_pl, y_box_iou_pl, y_center_delta_pl, y_dims_delta_pl, \
        y_orient_delta_pl

def get_model(boxpc, is_training, one_hot_vec, use_one_hot_vec=False, bn_decay=None, c=None):
    
    end_points = { 'class_ids' : tf.cast(tf.argmax(one_hot_vec, axis=1), dtype=tf.int32) }
    box_reg, pc = boxpc
    delta_dims = 3 + 3 + 1

    if not use_one_hot_vec: one_hot_vec = None
    
    # Predict if the box fits the point cloud or not + Delta term to correct the box
    output, feats = semisup_models.box_pc_mask_features_model(box_reg, pc, 
                    None, 2 + delta_dims, is_training, end_points=end_points, 
                    reuse=False, bn_for_output=False, one_hot_vec=one_hot_vec, 
                    norm_box2D=None, bn_decay=bn_decay, 
                    c=c, scope='box_pc_mask_model')
    boxpc_fit_logits = output[:,-2:]
    logits_for_weigh = tf.nn.softmax(boxpc_fit_logits)[:,1]
    pred_boxpc_fit = tf.cast(tf.nn.softmax(boxpc_fit_logits)[:,1] > 0.5, tf.int32)

    logits_for_weigh = tf.stop_gradient(logits_for_weigh) if c.BOXPC_STOP_GRAD_OF_CLS_VIA_DELTA \
                       else logits_for_weigh
    end_points['boxpc_feats_dict'] = feats                       
    end_points['boxpc_fit_logits'] = boxpc_fit_logits
    end_points['pred_boxpc_fit'] = pred_boxpc_fit
    end_points['logits_for_weigh'] = logits_for_weigh

    # Delta term
    boxpc_delta_center = output[:,0:3] # (B,3)
    boxpc_delta_size   = output[:,3:6] # (B,3)
    boxpc_delta_angle  = output[:,6]   # (B,)
    # Weigh the predictions by the cls confidence
    # (1 - logits) because if logits close to 1 (we are confident that the box fits), then
    # we do not want as much delta box to be applied
    if c.BOXPC_WEIGH_DELTA_PRED_BY_CLS_CONF:
        weigh_delta = 1. - logits_for_weigh
        boxpc_delta_center = boxpc_delta_center * tf_util.tf_expand_tile(weigh_delta, axis=1, tile=[1,3])
        boxpc_delta_size   = boxpc_delta_size * tf_util.tf_expand_tile(weigh_delta, axis=1, tile=[1,3])
        boxpc_delta_angle  = boxpc_delta_angle * weigh_delta
    end_points['boxpc_delta_center'] = boxpc_delta_center
    end_points['boxpc_delta_size'] = boxpc_delta_size
    end_points['boxpc_delta_angle'] = boxpc_delta_angle

    pred_delta_box = (boxpc_delta_center, boxpc_delta_size, boxpc_delta_angle)
    pred = (boxpc_fit_logits, pred_delta_box)

    return pred, end_points

##############################################################################
# Loss functions
##############################################################################

def get_loss(pred, labels, end_points, reduce_loss=True, c=None):
    
    # Box fit classification loss
    logits, _ = pred
    y_box_iou, _ = labels
    boxpc_cls_losses = get_boxpc_cls_loss(logits, y_box_iou, end_points, 
                                          reduce_loss=False, c=c)
    boxpc_cls_loss = tf.reduce_mean(boxpc_cls_losses)
    tf.summary.scalar('Strong_Loss/boxpc_cls_loss', boxpc_cls_loss)

    # Box fit delta regression loss
    boxpc_delta_losses = get_boxpc_delta_loss(pred, labels, end_points, 
                                              reduce_loss=False, c=c)
    boxpc_delta_loss = tf.reduce_mean(boxpc_delta_losses)
    tf.summary.scalar('Strong_Loss/boxpc_delta_loss', boxpc_delta_loss)

    total_losses = c.BOXPC_WEIGHT_CLS * boxpc_cls_losses + \
                   c.BOXPC_WEIGHT_DELTA * boxpc_delta_losses
    
    if reduce_loss:
        return tf.reduce_mean(total_losses)
    else:
        return total_losses

def get_boxpc_cls_loss(logits, y_box_iou, end_points, reduce_loss=True, c=None):

    # Classification losses
    classify_as_boxpc_fit = tf.one_hot(tf.cast(y_box_iou > c.BOXPC_FIT_BOUNDS[0], tf.int32), 
                                       depth=2)
    boxpc_cls_losses = tf.nn.softmax_cross_entropy_with_logits(labels=classify_as_boxpc_fit, 
                                                               logits=logits)

    if reduce_loss:
        return tf.reduce_mean(boxpc_cls_losses)
    else:
        return boxpc_cls_losses

def get_boxpc_delta_loss(pred, labels, end_points, reduce_loss=True, c=None):

    logits, pred_delta_box = pred
    y_box_iou, y_delta_box = labels
    boxpc_delta_center, boxpc_delta_size, boxpc_delta_angle = pred_delta_box
    boxpc_y_delta_center, boxpc_y_delta_size, boxpc_y_delta_angle = y_delta_box

    # Losses for prediction wrong deltas
    if c.BOXPC_DELTA_LOSS_TYPE == 'huber':
        boxpc_delta_center_losses = tf.losses.huber_loss(boxpc_y_delta_center, boxpc_delta_center, 
                                                         reduction=tf.losses.Reduction.NONE)
        boxpc_delta_size_losses = tf.losses.huber_loss(boxpc_y_delta_size, boxpc_delta_size, 
                                                       reduction=tf.losses.Reduction.NONE)
        boxpc_delta_angle_losses = tf.losses.huber_loss(boxpc_y_delta_angle, boxpc_delta_angle, 
                                                        reduction=tf.losses.Reduction.NONE)
    elif c.BOXPC_DELTA_LOSS_TYPE == 'mse':
        boxpc_delta_center_losses = tf.losses.mean_squared_error(boxpc_y_delta_center, 
                                    boxpc_delta_center, reduction=tf.losses.Reduction.NONE)
        boxpc_delta_size_losses = tf.losses.mean_squared_error(boxpc_y_delta_size, 
                                  boxpc_delta_size, reduction=tf.losses.Reduction.NONE)
        boxpc_delta_angle_losses = tf.losses.mean_squared_error(boxpc_y_delta_angle, 
                                   boxpc_delta_angle, reduction=tf.losses.Reduction.NONE)

    assert(not (c.BOXPC_WEIGH_DELTA_LOSS_BY_CLS_CONF and c.BOXPC_WEIGH_DELTA_LOSS_BY_CLS_GT))
    # Weigh the losses by the cls confidence/GT
    weigh_loss = 1.
    # (1 - logits) because if logits close to 1 (we are confident that the box fits), then
    # we do not want as much delta box loss to be applied
    logits_for_weigh = end_points['logits_for_weigh']
    if c.BOXPC_WEIGH_DELTA_LOSS_BY_CLS_CONF:
        weigh_loss = 1. - logits_for_weigh
    if c.BOXPC_WEIGH_DELTA_LOSS_BY_CLS_GT:
        weigh_loss = 1. - y_box_iou
    boxpc_delta_center_losses = tf.reduce_mean(boxpc_delta_center_losses, axis=1) * weigh_loss
    boxpc_delta_size_losses   = tf.reduce_mean(boxpc_delta_size_losses, axis=1) * weigh_loss
    boxpc_delta_angle_losses  = boxpc_delta_angle_losses * weigh_loss
    tf.summary.scalar('Strong_Loss/boxpc_delta_center_losses', tf.reduce_mean(boxpc_delta_center_losses))
    tf.summary.scalar('Strong_Loss/boxpc_delta_size_losses', tf.reduce_mean(boxpc_delta_size_losses))
    tf.summary.scalar('Strong_Loss/boxpc_delta_angle_losses', tf.reduce_mean(boxpc_delta_angle_losses))

    assert(c.BOXPC_WEIGHT_DELTA_CENTER_PERCENT + \
           c.BOXPC_WEIGHT_DELTA_SIZE_PERCENT + \
           c.BOXPC_WEIGHT_DELTA_ANGLE_PERCENT == 1)
    boxpc_delta_losses = c.BOXPC_WEIGHT_DELTA_CENTER_PERCENT * boxpc_delta_center_losses + \
                         c.BOXPC_WEIGHT_DELTA_SIZE_PERCENT * boxpc_delta_size_losses + \
                         c.BOXPC_WEIGHT_DELTA_ANGLE_PERCENT * boxpc_delta_angle_losses

    if reduce_loss:
        return tf.reduce_mean(boxpc_delta_losses)
    else:
        return boxpc_delta_losses

def huber_loss(error, delta, reduce_loss=True, scope='huber_loss'):

    with tf.variable_scope(scope) as sc:
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = (abs_error - quadratic)
        losses = 0.5 * quadratic**2 + delta * linear
        if reduce_loss:
            losses = tf.reduce_mean(losses)
        return losses

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