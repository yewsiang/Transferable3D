
""" Test on datasets with frustum proposals from RGB detections / Ground Truth.

Author: Tang Yew Siang
Date: July 2019
"""

import os
import sys
import importlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '../sunrgbd_data'))
sys.path.append(os.path.join(ROOT_DIR, '../models')) # fpn/models

import numpy as np
import tensorflow as tf

import tf_util
import roi_seg_box3d_dataset
from utils import save_zipped_pickle
from roi_seg_box3d_dataset import NUM_SIZE_CLUSTER, NUM_HEADING_BIN

def get_scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string
    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.
    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )

def load_variable_scopes_from_ckpt(restore_var_scopes, var_scope_names_in_ckpt, sess, ckpt_file):
    """
    Put var_scope_names_in_ckpt as restore_var_scopes if the scope is saved as the same name as the current
    instantiation.
    """
    var_list = {}
    for restore_var_scope, variable_scope_name_in_ckpt in zip(restore_var_scopes, var_scope_names_in_ckpt):
        restore_vars = get_scope_vars(restore_var_scope, trainable_only=False)
        for restore_var in restore_vars:
            original_var_name = restore_var.op.name.replace(restore_var_scope + '/', variable_scope_name_in_ckpt)
            var_list[original_var_name] = restore_var

    load_saver = tf.train.Saver(var_list=var_list)
    load_saver.restore(sess, ckpt_file)

def get_model(batch_size, num_point, num_channel, use_oracle_mask=False):

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            is_training_pl = tf.placeholder(tf.bool, shape=())

            pc_pl, bg_pc_pl, img_pl, one_hot_vec_pl, y_seg_pl, y_centers_pl, y_orient_cls_pl, \
            y_orient_reg_pl, y_dims_cls_pl, y_dims_reg_pl, R0_rect_pl, P_pl, Rtilt_pl, K_pl, \
            rot_frust_pl, box2D_pl, img_dim_pl, is_data_2D_pl = \
                MODEL.placeholder_inputs(batch_size, num_point, num_channel)
            labels = (y_seg_pl, y_centers_pl, y_orient_cls_pl, y_orient_reg_pl, y_dims_cls_pl, \
                      y_dims_reg_pl, R0_rect_pl, P_pl, Rtilt_pl, K_pl, rot_frust_pl, box2D_pl, \
                      img_dim_pl, is_data_2D_pl)
            norm_box2D = tf_util.tf_normalize_2D_bboxes(box2D_pl, img_dim_pl)
            oracle_mask = y_seg_pl if use_oracle_mask else None

            pred, end_points = MODEL.get_semi_model(pc_pl, bg_pc_pl, img_pl, one_hot_vec_pl, 
                                                    is_training_pl, norm_box2D=norm_box2D,
                                                    use_one_hot=FLAGS.use_one_hot, 
                                                    oracle_mask=oracle_mask, c=FLAGS)
            logits = pred[0]

            # ================================ BOXPC MODEL ================================
            prefix                                   = 'F_'
            FLAGS.SEMI_REFINE_USING_BOXPC_DELTA_NUM  = int(FLAGS.refine)
            FLAGS.SEMI_WEIGH_BOXPC_DELTA_DURING_TEST = False
            FLAGS.BOX_PC_MASK_REPRESENTATION         = 'A'

            import boxpc_sunrgbd
            is_training_D = tf.squeeze(tf.zeros(1, dtype=tf.bool))
            curr_box = end_points[prefix+'pred_box_reg']
            curr_center_reg, curr_size_reg, curr_angle_reg = curr_box
            
            # Compute the delta_box for a given (Box, PC) pair and add these
            # delta terms repeatedly. The final box terms are represented by
            # the 'F2_' terms.
            boxpc_fit_prob     = None
            total_delta_center = tf.zeros_like(curr_center_reg)
            total_delta_angle  = tf.zeros_like(curr_angle_reg)
            total_delta_size   = tf.zeros_like(curr_size_reg)
            for i in range(FLAGS.SEMI_REFINE_USING_BOXPC_DELTA_NUM):
                reuse = None if i == 0 else True
                if FLAGS.mask_pc_for_boxpc:
                    mask = tf.cast(tf.argmax(logits, axis=2), tf.float32)
                    mask = tf_util.tf_expand_tile(mask, axis=2, tile=[1,1,num_channel])
                    fake_box_pc = (curr_box, pc_pl * mask)
                else:
                    fake_box_pc = (curr_box, pc_pl)

                with tf.variable_scope('D_boxpc_branch', reuse=reuse):
                    fake_boxpc_pred, fake_boxpc_ep = boxpc_sunrgbd.get_model(fake_box_pc, 
                                                     is_training_D, 
                                                     one_hot_vec=one_hot_vec_pl, 
                                                     use_one_hot_vec=False,
                                                     c=FLAGS)
                boxpc_fit_prob = tf.nn.softmax(fake_boxpc_ep['boxpc_fit_logits'])[:,1]
                
                # Final predicted box
                weight = 1 - fake_boxpc_ep['logits_for_weigh'] if \
                    FLAGS.SEMI_WEIGH_BOXPC_DELTA_DURING_TEST else \
                    tf.ones_like(fake_boxpc_ep['logits_for_weigh'])
                weight_exp = tf_util.tf_expand_tile(weight, axis=1, tile=[1,3])
                delta_center = fake_boxpc_ep['boxpc_delta_center'] * weight_exp
                delta_angle  = fake_boxpc_ep['boxpc_delta_angle'] * weight
                delta_size   = fake_boxpc_ep['boxpc_delta_size'] * weight_exp
                curr_center_reg, curr_size_reg, curr_angle_reg = curr_box
                refined_center_reg = curr_center_reg - delta_center
                refined_angle_reg  = curr_angle_reg - delta_angle
                refined_size_reg   = curr_size_reg - delta_size
                curr_box = (refined_center_reg, refined_size_reg, refined_angle_reg)

                total_delta_center = total_delta_center + delta_center
                total_delta_angle = total_delta_angle + delta_angle
                total_delta_size = total_delta_size + delta_size

            F2_center            = end_points[prefix+'center'] - total_delta_center
            F2_heading_scores    = end_points[prefix+'heading_scores']
            F2_heading_residuals = end_points[prefix+'heading_residuals'] - \
                                   tf_util.tf_expand_tile(total_delta_angle, axis=1, tile=[1,12])
            F2_size_scores       = end_points[prefix+'size_scores']
            F2_size_residuals    = end_points[prefix+'size_residuals'] - \
                                   tf_util.tf_expand_tile(total_delta_size, axis=1, tile=[1,10,1])

            end_points.update({ 'boxpc_fit_prob'       : boxpc_fit_prob,
                                'F2_center'            : F2_center,
                                'F2_heading_scores'    : F2_heading_scores,
                                'F2_heading_residuals' : F2_heading_residuals,
                                'F2_size_scores'       : F2_size_scores,
                                'F2_size_residuals'    : F2_size_residuals })

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)

        ops = { 'pc_pl'           : pc_pl,
                'one_hot_vec_pl'  : one_hot_vec_pl,
                'y_seg_pl'        : y_seg_pl,
                'y_centers_pl'    : y_centers_pl,
                'y_orient_cls_pl' : y_orient_cls_pl,
                'y_orient_reg_pl' : y_orient_reg_pl,
                'y_dims_cls_pl'   : y_dims_cls_pl,
                'y_dims_reg_pl'   : y_dims_reg_pl,
                'R0_rect_pl'      : R0_rect_pl,
                'P_pl'            : P_pl,
                'Rtilt_pl'        : Rtilt_pl,
                'K_pl'            : K_pl,
                'rot_frust_pl'    : rot_frust_pl,
                'box2D_pl'        : box2D_pl,
                'img_dim_pl'      : img_dim_pl,
                'is_training_pl'  : is_training_pl,
                'logits'          : logits,
                'end_points'      : end_points }
        return sess, ops

def softmax(x):
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def inference(sess, ops, pc, one_hot_vec, batch_size, prefix='', 
    use_boxpc_fit_prob=False, oracle_mask=None):
    ''' pc: BxNx3 array, Bx3 array, return BxN pred and Bx3 centers '''
    assert pc.shape[0]%batch_size == 0
    num_batches       = pc.shape[0]/batch_size
    boxpc_fit_prob    = np.zeros((pc.shape[0],))
    logits            = np.zeros((pc.shape[0], pc.shape[1], 2))
    centers           = np.zeros((pc.shape[0], 3))
    heading_logits    = np.zeros((pc.shape[0],NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0],NUM_HEADING_BIN))
    size_logits       = np.zeros((pc.shape[0],NUM_SIZE_CLUSTER))
    size_residuals    = np.zeros((pc.shape[0],NUM_SIZE_CLUSTER,3))
    # score that indicates confidence in 3d box prediction (mask logits+heading+size); no confidence for the center...
    scores            = np.zeros((pc.shape[0],)) 
   
    ep = ops['end_points'] 
    for i in range(num_batches):
        feed_dict = { ops['pc_pl']          : pc[i*batch_size:(i+1)*batch_size,...],
                      ops['one_hot_vec_pl'] : one_hot_vec[i*batch_size:(i+1)*batch_size,:],
                      ops['is_training_pl'] : False }
        if not oracle_mask is None:
            feed_dict.update({ ops['y_seg_pl'] : oracle_mask })

        run_ops = [ ops['logits'], 
                    ep[prefix+'center'], 
                    ep[prefix+'heading_scores'], 
                    ep[prefix+'heading_residuals'], 
                    ep[prefix+'size_scores'], 
                    ep[prefix+'size_residuals']]

        if use_boxpc_fit_prob:
            run_ops.append(ep['boxpc_fit_prob'])
            batch_logits, batch_centers, batch_heading_scores, batch_heading_residuals, \
            batch_size_scores, batch_size_residuals, batch_boxpc_fit_prob = \
                sess.run(run_ops, feed_dict=feed_dict)
            boxpc_fit_prob[i*batch_size:(i+1)*batch_size,...] = batch_boxpc_fit_prob
        else:
            batch_logits, batch_centers, batch_heading_scores, batch_heading_residuals, \
            batch_size_scores, batch_size_residuals = \
                sess.run(run_ops, feed_dict=feed_dict)

        logits[i*batch_size:(i+1)*batch_size,...]            = batch_logits
        centers[i*batch_size:(i+1)*batch_size,...]           = batch_centers
        heading_logits[i*batch_size:(i+1)*batch_size,...]    = batch_heading_scores
        heading_residuals[i*batch_size:(i+1)*batch_size,...] = batch_heading_residuals
        size_logits[i*batch_size:(i+1)*batch_size,...]       = batch_size_scores
        size_residuals[i*batch_size:(i+1)*batch_size,...]    = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:,:,1] # BxN
        batch_seg_mask = np.argmax(batch_logits, 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        mask_mean_prob = mask_mean_prob / (np.sum(batch_seg_mask,1) + 1) # B, (YS)
        # mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
        heading_prob   = np.max(softmax(batch_heading_scores),1) # B
        size_prob      = np.max(softmax(batch_size_scores),1) # B,
        if use_boxpc_fit_prob:
            batch_scores   = np.log(batch_boxpc_fit_prob + 0.01) + np.log(mask_mean_prob + 0.01) + np.log(heading_prob + 0.01) + np.log(size_prob + 0.01) # (YS)
        else:
            batch_scores   = np.log(mask_mean_prob + 0.01) + np.log(heading_prob + 0.01) + np.log(size_prob + 0.01) # (YS)
            # batch_scores   = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        scores[i*batch_size:(i+1)*batch_size] = batch_scores 
        # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls    = np.argmax(size_logits, 1) # B
    
    pred_seg        = np.argmax(logits, 2)
    pred_orient_cls = heading_cls
    pred_orient_reg = np.array([heading_residuals[i,heading_cls[i]] for i in range(pc.shape[0])])
    pred_dims_cls   = size_cls
    pred_dims_reg   = np.vstack([size_residuals[i,size_cls[i],:] for i in range(pc.shape[0])])

    return pred_seg, centers, pred_orient_cls, pred_orient_reg, pred_dims_cls, pred_dims_reg, scores

def write_detection_results(result_dir, test_classes, id_list, type_list, box2d_list, center_list, heading_cls_list, heading_res_list, size_cls_list, size_res_list, rot_angle_list, score_list):
    """
    Output to N_class number of files in result_dir, where each file is result_dir/cls_1.txt, ...,
    result_dir/cls_N_class.txt.
    """
    assert(result_dir is not None)

    # Open files according to class
    cls_files = {}
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    for test_class in test_classes:
        cls_file = os.path.join(result_dir, test_class + '_pred.txt')
        file = open(cls_file, 'w')
        cls_files[test_class] = file 

    # Put prediction result into files
    for i in range(len(center_list)):
        idx      = id_list[i]
        box2d    = box2d_list[i]
        cls_name = type_list[i]
        score    = score_list[i]
        file     = cls_files[cls_name]
        h,w,l,tx,ty,tz,ry = roi_seg_box3d_dataset.from_prediction_to_label_format(center_list[i], 
            heading_cls_list[i], heading_res_list[i], size_cls_list[i], size_res_list[i], rot_angle_list[i])

        # Note that idx is placed at the front
        output_str = '%d %s -1 -1 -10 %f %f %f %f %f %f %f %f %f %f %f %f\n' % \
            (idx, cls_name, box2d[0], box2d[1], box2d[2], box2d[3],
             h, w, l, tx, ty, tz, ry, score)
        file.write(output_str)

    # Close files
    for _, file in cls_files.items():
        file.close()

def write_gt_results(result_dir, test_classes, test_dataset):
    assert(result_dir is not None)

    # Open files according to class
    cls_files = {}
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    for test_class in test_classes:
        cls_file = os.path.join(result_dir, test_class + '_gt.txt')
        file = open(cls_file, 'w')
        cls_files[test_class] = file 

    # Put prediction result into files
    for i in range(len(test_dataset)):
        idx      = test_dataset.idx_l[i]
        box2d    = test_dataset.box2d_l[i]
        cls_name = test_dataset.cls_type_l[i]
        center   = test_dataset.get_box3d_center(i)
        size     = test_dataset.size_l[i]
        heading  = test_dataset.heading_l[i]
        file     = cls_files[cls_name]
        l, w, h    = size
        tx, ty, tz = center

        # Note that idx is placed at the front
        output_str = '%d %s -1 -1 -10 %f %f %f %f %f %f %f %f %f %f %f\n' % \
            (idx, cls_name, box2d[0], box2d[1], box2d[2], box2d[3],
             h, w, l, tx, ty, tz, heading)
        file.write(output_str)

    # Close files
    for _, file in cls_files.items():
        file.close()

def fill_files(output_dir, to_fill_filename_list):
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

def main_batch_from_rgb_detection(test_dataset, test_classes, num_class, num_point, num_channel, 
    prefix='', semi_type=None, use_boxpc_fit_prob=False, use_oracle_mask=False, sess_ops=None, output_filename=None, 
    result_dir=None, verbose=False):
    ps_list = []
    image_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    cls_list = []
    file_num_list = []
    box2d_list = []

    test_idxs = np.arange(0, len(test_dataset))
    batch_size = 32
    num_batches = int((len(test_dataset)+batch_size-1)/batch_size)
    
    batch_data_to_feed = np.zeros((batch_size, num_point, num_channel))
    batch_one_hot_to_feed = np.zeros((batch_size, num_class))
    batch_oracle_mask_to_feed = np.zeros((batch_size, num_point))
    if sess_ops is None:
        sess, ops = get_model(batch_size=batch_size, 
                              num_point=num_point, 
                              num_channel=num_channel,
                              use_oracle_mask=use_oracle_mask)
    else:
        sess, ops = sess_ops

    idx = 0
    for batch_idx in range(num_batches):
        if verbose and batch_idx % 10 == 0: print(batch_idx)
        start_idx = batch_idx * batch_size
        end_idx = min(len(test_dataset), (batch_idx+1) * batch_size)
        cur_batch_size = end_idx - start_idx

        batch_data, batch_image, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec, \
        batch_oracle_y_seg = \
            test_dataset.get_batch(test_idxs, start_idx, end_idx, num_point, num_channel, 
                                   from_rgb_detection=True)
        
        batch_data_to_feed[0:cur_batch_size,...] = batch_data
        batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec
        if use_oracle_mask:
            batch_oracle_mask_to_feed[0:cur_batch_size,...] = batch_oracle_y_seg
        else:
            batch_oracle_mask_to_feed = None
        batch_output, batch_center_pred, batch_hclass_pred, batch_hres_pred, batch_sclass_pred, \
        batch_sres_pred, batch_scores = inference(sess, ops, batch_data_to_feed, \
                                                  batch_one_hot_to_feed, 
                                                  use_boxpc_fit_prob=use_boxpc_fit_prob,
                                                  oracle_mask=batch_oracle_mask_to_feed,
                                                  batch_size=batch_size,
                                                  prefix=prefix)
    
        for i in range(cur_batch_size):
            ps_list.append(batch_data[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            # Combine 3D BOX score and 2D RGB detection score
            # score_list.append(batch_scores[i] + np.log(batch_rgb_prob[i])) 
            score_list.append(batch_rgb_prob[i]) # 2D RGB detection score

            cls_list.append(np.argmax(batch_one_hot_vec[i]))
            file_num_list.append(test_dataset.idx_l[idx])
            box2d_list.append(test_dataset.box2d_l[idx])
            idx += 1

    predictions = [ps_list, None, segp_list, center_list, heading_cls_list, \
                   heading_res_list, size_cls_list, size_res_list, rot_angle_list, \
                   score_list, cls_list, file_num_list, box2d_list, None]

    if output_filename is not None:
        save_zipped_pickle(predictions, output_filename) 

    if result_dir is not None:
        # Write detection results for MATLAB evaluation
        write_detection_results(result_dir, test_classes, test_dataset.idx_l, test_dataset.cls_type_l, \
            test_dataset.box2d_l, center_list, heading_cls_list, heading_res_list, size_cls_list, \
            size_res_list, rot_angle_list, score_list)

    return predictions

def main_batch(test_dataset, test_classes, num_class, num_point, num_channel, 
    prefix='', semi_type=None, use_boxpc_fit_prob=False, sess_ops=None, output_filename=None, 
    result_dir=None, verbose=False):
    ps_list = []
    seg_gt_list = []
    seg_pred_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    cls_list = []
    file_num_list = []
    box2d_list = []
    box3d_list = []

    test_idxs = np.arange(0, len(test_dataset))
    batch_size = 32
    num_batches = int((len(test_dataset)+batch_size-1)/batch_size)

    batch_data_to_feed = np.zeros((batch_size, num_point, num_channel))
    batch_one_hot_to_feed = np.zeros((batch_size, num_class))
    if sess_ops is None:
        sess, ops = get_model(batch_size=batch_size, num_point=num_point, num_channel=num_channel)
    else:
        sess, ops = sess_ops
    
    idx = 0
    iou_sum = 0
    for batch_idx in range(num_batches):
        if verbose and batch_idx % 10 == 0: print(batch_idx)
        start_idx = batch_idx * batch_size
        end_idx = min(len(test_dataset), (batch_idx+1) * batch_size)
        cur_batch_size = end_idx - start_idx
        
        batch_data, batch_image, batch_label, batch_center, batch_hclass, batch_hres, \
        batch_sclass, batch_sres, batch_box2d, batch_rtilt, batch_k, batch_rot_angle, \
        batch_img_dims, batch_one_hot_vec = \
            test_dataset.get_batch(test_idxs, start_idx, end_idx, num_point, num_channel)

        batch_data_to_feed[0:cur_batch_size,...] = batch_data
        batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec
        batch_output, batch_center_pred, batch_hclass_pred, batch_hres_pred, batch_sclass_pred, \
        batch_sres_pred, batch_scores = inference(sess, ops, batch_data_to_feed, \
                                                  batch_one_hot_to_feed, 
                                                  use_boxpc_fit_prob=use_boxpc_fit_prob,
                                                  batch_size=batch_size,
                                                  prefix=prefix)
        
        # The point cloud has been duplicated, we only want to calculate the IOU
        # on the points that are not duplicated
        # correct_cnt += np.sum(batch_output==batch_label)
        for i, (point_cloud, seg_pred, seg_gt) in enumerate(zip(batch_data, batch_output, batch_label)):
            if start_idx + i >= len(test_dataset.idx_l):
                continue
            _, unique_idx = np.unique(point_cloud, axis=0, return_index=True)
            y_pred = seg_pred[unique_idx]
            y_true = seg_gt[unique_idx]
            iou = float(np.sum(y_pred & y_true)) / (np.sum(y_pred | y_true) + 1)
            iou_sum += iou

        for i in range(cur_batch_size):
            ps_list.append(batch_data[i,...])
            seg_gt_list.append(batch_label[i,...])
            seg_pred_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])
            cls_list.append(np.argmax(batch_one_hot_vec[i]))
            file_num_list.append(test_dataset.idx_l[idx])
            box2d_list.append(test_dataset.box2d_l[idx])
            box3d_list.append(test_dataset.box3d_l[idx])
            idx += 1

    print("Mean segmentation IOU: %f" % (iou_sum / len(test_dataset.idx_l)))
    predictions = [ps_list, seg_gt_list, seg_pred_list, center_list, heading_cls_list, 
                   heading_res_list, size_cls_list, size_res_list, rot_angle_list, \
                   score_list, cls_list, file_num_list, box2d_list, box3d_list]

    if output_filename is not None:
        save_zipped_pickle(predictions, output_filename) 

    if result_dir is not None:
        # Write detection results for MATLAB evaluation
        write_detection_results(result_dir, test_classes, test_dataset.idx_l, test_dataset.cls_type_l, \
            test_dataset.box2d_l, center_list, heading_cls_list, heading_res_list, size_cls_list, \
            size_res_list, rot_angle_list, score_list)

    return predictions

if __name__=='__main__':

    from config import cfg
    cfg.add_argument('--test', type=str, required=True, nargs='+', help='Classes to test with. Specify AB to test all classes, A or B for specific subsets of classes corresponding to paper.')
    cfg.add_argument('--semi_type', type=str, choices=['A', 'F'], help='Model + Loss function used for semisup model.')
    cfg.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    cfg.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
    cfg.add_argument('--model', required=True, help='Model name [default: model]')
    cfg.add_argument('--model_path', required=True, help='model checkpoint file path')
    cfg.add_argument('--boxpc_model_path', default=None, help='boxpc model checkpoint file path [default: None]')
    cfg.add_argument('--pred_prefix', required=True, help='prefix to add for predictions')
    cfg.add_argument('--refine', default=None, help='Number of times to refine by')
    cfg.add_argument('--output', required=True, help='output filename]')
    cfg.add_argument('--data_path', default=None, help='data path [default: None]')
    cfg.add_argument('--no_rgb', action='store_true', help='No RGB information.')
    cfg.add_argument('--mask_pc_for_boxpc', action='store_true', help='Mask the PC before giving to BoxPC network.')
    cfg.add_argument('--use_boxpc', action='store_true', help='Use BoxPC model (relevant to frustum_pointnets_v1 or models that do not already have BoxPC model).')
    cfg.add_argument('--use_boxpc_fit_prob', action='store_true', help='Use BoxPC model\'s boxpc_fit_prob in inference score.')
    cfg.add_argument('--use_one_hot', action='store_true', help='Use one hot vector if model used it during training.')
    cfg.add_argument('--from_rgb_detection', action='store_true', help='test from data file from rgb detection.')
    cfg.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
    FLAGS = cfg.parse_special_args()

    FLAGS.SEMI_MODEL = FLAGS.semi_type
    if FLAGS.use_boxpc:
        assert(FLAGS.boxpc_model_path is not None)

    ALL_CLASSES = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
    if FLAGS.test[0] == 'AB':
        WHITE_LIST = ALL_CLASSES
    elif FLAGS.test[0] == 'A':
        WHITE_LIST = ['bed','chair','toilet','desk','bathtub']
    elif FLAGS.test[0] == 'B':
        WHITE_LIST = ['table','sofa','dresser','night_stand','bookshelf']
    else:
        # Make sure all of the classes specified are valid
        WHITE_LIST = []
        for cls_name in FLAGS.test:
            if not (cls_name in ALL_CLASSES): assert(False)
            WHITE_LIST.append(cls_name)


    MODEL_PATH = FLAGS.model_path
    GPU_INDEX = FLAGS.gpu
    NUM_POINT = FLAGS.num_point
    MODEL = importlib.import_module(FLAGS.model) # import network module
    NUM_CHANNEL = 3 if FLAGS.no_rgb else 6

    num_classes = len(ALL_CLASSES)
    TEST_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(WHITE_LIST, npoints=NUM_POINT, 
                                                          split='training', rotate_to_center=True, \
                                                          overwritten_data_path=FLAGS.data_path, \
                                                          from_rgb_detection=FLAGS.from_rgb_detection, \
                                                          one_hot=True) # Always take one hot from dataset but may not use it
    print('Test Dataset size: %d' % len(TEST_DATASET))

    # Detections
    print('3D Object Detection...')
    if FLAGS.from_rgb_detection:
        main_batch_from_rgb_detection(TEST_DATASET, WHITE_LIST, num_classes, NUM_POINT, NUM_CHANNEL, 
                   prefix=FLAGS.pred_prefix,
                   semi_type=FLAGS.semi_type, 
                   use_boxpc_fit_prob=FLAGS.use_boxpc_fit_prob, 
                   use_oracle_mask=FLAGS.use_oracle_mask,
                   output_filename=os.path.join('predictions3D', FLAGS.output+'.pickle'), 
                   result_dir=os.path.join('predictions3D', FLAGS.output),
                   verbose=True)
    else:
        main_batch(TEST_DATASET, WHITE_LIST, num_classes, NUM_POINT, NUM_CHANNEL, 
                   prefix=FLAGS.pred_prefix,
                   semi_type=FLAGS.semi_type, 
                   use_boxpc_fit_prob=FLAGS.use_boxpc_fit_prob, 
                   output_filename=os.path.join('predictions3D', FLAGS.output+'.pickle'), 
                   result_dir=os.path.join('predictions3D', FLAGS.output),
                   verbose=True)

    print('Finished testing on %d data points' % len(TEST_DATASET))
