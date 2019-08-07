
""" Train Semi-supervised model.

Author: Tang Yew Siang
Date: July 2019
"""

import os
import sys
import importlib
from datetime import datetime
from os.path import join as pjoin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # sunrgbd/sunrgbd_detection
ROOT_DIR = os.path.dirname(BASE_DIR)                   # sunrgbd
MODEL_DIR = pjoin(os.path.dirname(ROOT_DIR), 'models') # fpn/models
sys.path.append(BASE_DIR) # model
sys.path.append(MODEL_DIR)

import numpy as np
import tensorflow as tf

import tf_util
import weak_losses
import roi_seg_box3d_dataset, roi_semi_dataset
from config import cfg
from timer import Timer

cfg.add_argument('--train_data', type=str, required=True, choices=['train_mini', 'train_aug5x', 'trainval_aug5x'])
cfg.add_argument('--train_data3D_keep_prob', type=float, default=1, help='Percentage of 3D data to keep randomly. 1 means no data is dropped.')
cfg.add_argument('--add3D_for_classes2D_prob', type=float, default=-1, help='Percentage of 3D data added for 2D classes. 0 means no data is added.')
cfg.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
cfg.add_argument('--model', default='semisup_v1_sunrgbd', help='Model name [default: model]')
cfg.add_argument('--log_dir', default='log', help='Log dir [default: log]')
cfg.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
cfg.add_argument('--max_epoch', type=int, default=31, help='Epoch to run [default: 51]')
cfg.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
cfg.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
cfg.add_argument('--learning_rate_D', type=float, default=0.001, help='Initial learning rate for Discriminator [default: 0.001]')
cfg.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
cfg.add_argument('--optimizer', default='adam', help='adam or momentum or sgd [default: adam]')
cfg.add_argument('--optimizer_D', default='sgd', help='adam or momentum or sgd for Discriminator [default: sgd]')
cfg.add_argument('--decay_step', type=int, default=800000, help='Decay step for lr decay [default: 200000]')
cfg.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
cfg.add_argument('--use_mini', action='store_true', help='Use mini dataset for training data')
cfg.add_argument('--use_one_hot', action='store_true', help='Use one hot vector during training')
cfg.add_argument('--use_one_hot_boxpc', action='store_true', help='Use one hot vector for BoxPC network during training')
cfg.add_argument('--no_aug', action='store_true', help='Do not augment data during training')
cfg.add_argument('--no_rgb', action='store_true', help='Only use XYZ for training')
cfg.add_argument('--init_class_ag_path', default=None, help='Model parameters for class agnostic branch e.g. log/model.ckpt [default: None]')
cfg.add_argument('--init_boxpc_path', type=str, default=None, nargs='+', help='Model parameters for boxpc branch e.g. log/model.ckpt [default: None]')
cfg.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
FLAGS = cfg.parse_special_args()

# Parameters
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BASE_LEARNING_RATE_D = FLAGS.learning_rate_D
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
OPTIMIZER_D = FLAGS.optimizer_D
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
FLAGS.NUM_CHANNELS = 3 if FLAGS.no_rgb else 6
NUM_SEG_CLASSES = 2
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


# Classes
ALL_CLASSES = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
FLAGS.TRAIN_CLS = FLAGS.SUNRGBD_SEMI_TRAIN_CLS
FLAGS.TEST_CLS  = FLAGS.SUNRGBD_SEMI_TEST_CLS


# Files and logs
MODEL        = importlib.import_module(FLAGS.model) # import network module
TRAIN_FILE   = pjoin(BASE_DIR, 'train_semisup.py')
TRAIN_FILE2  = pjoin(BASE_DIR, 'train_semisup_adv.py')
MODEL_FILE   = pjoin(BASE_DIR, FLAGS.model + '.py')
MODELS_FILE  = pjoin(BASE_DIR, 'semisup_models.py')
WLOSSES_FILE = pjoin(MODEL_DIR, 'weak_losses.py')
CONFIG_FILE  = pjoin(MODEL_DIR, 'config.py')
LOG_DIR      = pjoin('experiments_adv', FLAGS.log_dir)
if not os.path.exists('experiments_adv'): os.mkdir('experiments_adv')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# Backup of relevant code
for file in [TRAIN_FILE, TRAIN_FILE2, MODEL_FILE, MODELS_FILE, WLOSSES_FILE, CONFIG_FILE]:
    os.system('cp %s %s' % (file, LOG_DIR))
LOG_FOUT = open(pjoin(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
T1 = Timer('1 Epoch')
T2 = Timer('%d D cycles' % FLAGS.SEMI_ADV_ITERS_FOR_D)


# Datasets
train_file = 'train_mini.zip.pickle' if FLAGS.use_mini else ('%s.zip.pickle' % FLAGS.train_data)
test_file  = 'train_mini.zip.pickle' if FLAGS.use_mini else 'test.zip.pickle'
aug_data   = False if FLAGS.no_aug else True
classes2D = list(set(FLAGS.TRAIN_CLS + FLAGS.TEST_CLS)) if FLAGS.SEMI_USE_LABELS2D_OF_CLASSES3D else FLAGS.TEST_CLS
TRAIN_DATASET = roi_semi_dataset.ROISemiDataset(
                classes3D=FLAGS.TRAIN_CLS, classes2D=classes2D,
                data3D_keep_prob=FLAGS.train_data3D_keep_prob, 
                add3D_for_classes2D_prob=FLAGS.add3D_for_classes2D_prob,
                npoints=NUM_POINT, rotate_to_center=True, random_flip=aug_data, random_shift=aug_data, 
                overwritten_data_path=pjoin('frustums',train_file))
print('Length of Train Dataset: (2D: %d, 3D: %d)' % \
    (TRAIN_DATASET.get_len_classes2D(), TRAIN_DATASET.get_len_classes3D()))

TEST_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(FLAGS.TEST_CLS, npoints=NUM_POINT, 
               split='val', rotate_to_center=True, 
               overwritten_data_path=pjoin('frustums',test_file), one_hot=True)
print('Length of Final Test Dataset : %d' % len(TEST_DATASET))


# For evaluation purposes
from sunrgbd_data import sunrgbd_object
from test_semisup import main_batch, main_batch_from_rgb_detection
from evaluate import evaluate_predictions, get_ap_info

SUNRGBD_DATASET_DIR = '/home/yewsiang/Transferable3D/dataset/mysunrgbd'
DATASET = sunrgbd_object(SUNRGBD_DATASET_DIR, 'training')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch, base_learning_rate):
    learning_rate = tf.train.exponential_decay(
                        base_learning_rate,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

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

def get_scope_vars_except_unwanted_scope(unwanted_scope, trainable_only=False):
    """
    Get variables outside a scope
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
    variables = tf.trainable_variables() if trainable_only else tf.global_variables()
    interested_vars = []
    for variable in variables:
        unwanted_scope_name = unwanted_scope if isinstance(unwanted_scope, str) else unwanted_scope.name
        if unwanted_scope_name not in variable.name.split('/'):
            interested_vars.append(variable)
    return interested_vars

def get_scope_vars_except_unwanted_scopes(unwanted_scopes, trainable_only=False):
    """
    Get variables outside a scope
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
    variables = tf.trainable_variables() if trainable_only else tf.global_variables()
    interested_vars = []
    for variable in variables:
        unwanted_scope_names = [(scope if isinstance(scope, str) else scope.name) for scope in unwanted_scopes]
        # for each scope of the current variable, is it contained within the list of unwanted scopes?
        is_unwanted_scope_name_in_variable_name = [(name in unwanted_scope_names) for name in variable.name.split('/')]
        # if there are no unwanted scopes within the current variable, then we want the variable
        if not any(is_unwanted_scope_name_in_variable_name):
            interested_vars.append(variable)
    return interested_vars

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

def add_ap_summary(ap, mean_ap, writer, step, prefix='AP/'):
    for i, classname in enumerate(ap.keys()):
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='%s%s' % (prefix, classname), simple_value=ap[classname])]), step)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='%sMean' % prefix, simple_value=mean_ap)]), step)

def add_adv_class_stats_summary(adv_cls_stats, writer, step):
    classes = adv_cls_stats.keys()
    classes.sort()
    precs, recalls, f1s, supports = [], [], [], []
    for cls_type in classes:
        prec, recall, f1, support = adv_cls_stats[cls_type]
        precs.append(prec)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Precision/%s' % cls_type, simple_value=prec)]), step)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Recall/%s' % cls_type, simple_value=recall)]), step)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='F1/%s' % cls_type, simple_value=f1)]), step)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Support/%s' % cls_type, simple_value=support)]), step)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Precision/Mean', simple_value=np.mean(precs))]), step)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Recall/Mean', simple_value=np.mean(recalls))]), step)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='F1/Mean', simple_value=np.mean(f1s))]), step)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Support/Mean', simple_value=np.mean(supports))]), step)

##############################################################################
# Training
##############################################################################

def train():
    # Print configurations
    log_string('\n\nCommand:\npython %s\n' % ' '.join(sys.argv))
    log_string(FLAGS.config_str)

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pc_pl, bg_pc_pl, img_pl, one_hot_vec_pl, y_seg_pl, y_centers_pl, y_orient_cls_pl, \
            y_orient_reg_pl, y_dims_cls_pl, y_dims_reg_pl, R0_rect_pl, P_pl, Rtilt_pl, K_pl, \
            rot_frust_pl, box2D_pl, img_dim_pl, is_data_2D_pl = \
                MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, FLAGS.NUM_CHANNELS)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # ================================== OPTIMIZERS ==================================
            # Note the global_step=batch parameter to minimize. That tells the optimizer to 
            # helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('Info/bn_decay', bn_decay)
            batch_D = tf.Variable(0)
            bn_decay_D = get_bn_decay(batch_D)
            tf.summary.scalar('Info/bn_decay_D', bn_decay_D)
            
            # Get training operator
            learning_rate = get_learning_rate(batch, BASE_LEARNING_RATE)
            tf.summary.scalar('Info/learning_rate', learning_rate)
            learning_rate_D = get_learning_rate(batch_D, BASE_LEARNING_RATE_D)
            tf.summary.scalar('Info/learning_rate_D', learning_rate_D)
            if OPTIMIZER == 'momentum':
                optimizer   = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer   = tf.train.AdamOptimizer(learning_rate)
            elif OPTIMIZER == 'sgd':
                optimizer   = tf.train.GradientDescentOptimizer(learning_rate)
            if OPTIMIZER_D == 'momentum':
                optimizer_D = tf.train.MomentumOptimizer(learning_rate_D, momentum=MOMENTUM)
            elif OPTIMIZER_D == 'adam':
                optimizer_D = tf.train.AdamOptimizer(learning_rate_D)
            elif OPTIMIZER_D == 'sgd':
                optimizer_D = tf.train.GradientDescentOptimizer(learning_rate_D)

            # ==================================== MODEL ====================================
            # Get model and loss
            labels = (y_seg_pl, y_centers_pl, y_orient_cls_pl, y_orient_reg_pl, y_dims_cls_pl, \
                      y_dims_reg_pl, R0_rect_pl, P_pl, Rtilt_pl, K_pl, rot_frust_pl, box2D_pl, \
                      img_dim_pl, is_data_2D_pl)
            # NOTE: Only use ONE optimizer during each training to prevent batch + optimizer values 
            # getting affected
            norm_box2D = tf_util.tf_normalize_2D_bboxes(box2D_pl, img_dim_pl)
            pred, end_points = MODEL.get_semi_model(pc_pl, bg_pc_pl, img_pl, one_hot_vec_pl, 
                                     is_training_pl, use_one_hot=FLAGS.use_one_hot, 
                                     norm_box2D=norm_box2D, bn_decay=bn_decay, c=FLAGS)
            
            intraclsdims_train_classes = [(True if cls_type in FLAGS.TEST_CLS else False) for cls_type \
                in ALL_CLASSES] if FLAGS.SEMI_INTRACLSDIMS_ONLY_ON_2D_CLS else [True] * len(ALL_CLASSES)
            inactive_vol_train_classes = [(True if cls_type in FLAGS.TEST_CLS else False) for cls_type \
                in ALL_CLASSES] if FLAGS.WEAK_INACTIVE_VOL_ONLY_ON_2D_CLS else [True] * len(ALL_CLASSES)
            end_points.update({ 'intraclsdims_train_classes' : intraclsdims_train_classes,
                                'inactive_vol_train_classes' : inactive_vol_train_classes })
            log_string('\n  Train on 2D only    (Reprojection): %s\n' % str(FLAGS.WEAK_REPROJECTION_ONLY_ON_2D_CLS))
            log_string('\n  Train on 2D only    (Box PC Fit)  : %s\n' % str(FLAGS.SEMI_BOXPC_FIT_ONLY_ON_2D_CLS))
            log_string('\n  Classes to train on (Inactive vol): %s\n' % str(inactive_vol_train_classes))
            logits = pred[0]

            # ====================================== D MODEL ======================================
            import boxpc_sunrgbd
            y_box = (y_centers_pl, y_orient_cls_pl, y_orient_reg_pl, y_dims_cls_pl, y_dims_reg_pl)
            y_box_reg = boxpc_sunrgbd.convert_raw_y_box_to_reg_format(y_box, one_hot_vec_pl)
            real_box_pc = (y_box_reg, pc_pl)
            fake_box_pc = (end_points['F_pred_box_reg'], pc_pl)
            is_training_D = is_training_pl if FLAGS.SEMI_TRAIN_BOXPC_MODEL else tf.squeeze(tf.zeros(1, dtype=tf.bool))
            with tf.variable_scope('D_boxpc_branch', reuse=None):
                real_boxpc_pred, real_boxpc_ep = boxpc_sunrgbd.get_model(real_box_pc, is_training_D, 
                                                 one_hot_vec_pl, use_one_hot_vec=FLAGS.use_one_hot_boxpc, 
                                                 bn_decay=bn_decay, c=FLAGS)
            with tf.variable_scope('D_boxpc_branch', reuse=True):
                fake_boxpc_pred, fake_boxpc_ep = boxpc_sunrgbd.get_model(fake_box_pc, is_training_D, 
                                                 one_hot_vec_pl, use_one_hot_vec=FLAGS.use_one_hot_boxpc, 
                                                 bn_decay=bn_decay, c=FLAGS)
            logits_real = real_boxpc_ep['boxpc_fit_logits']
            logits_fake = fake_boxpc_ep['boxpc_fit_logits']
            D_loss = weak_losses.get_D_loss(logits_real, logits_fake, 
                                 loss_type='SOFTMAX',
                                 use_soft_noisy_labels_D=FLAGS.SEMI_ADV_SOFT_NOISY_LABELS_FOR_D,
                                 flip_labels_prob=FLAGS.SEMI_ADV_FLIP_LABELS_FOR_D_PROB,
                                 mask_real=tf.cast(1 - is_data_2D_pl, tf.float32), 
                                 mask_fake=None,
                                 scope='D_loss')
            trainable_D_vars = get_scope_vars('D_boxpc_branch', trainable_only=True)
            fake_boxpc_fit_prob = tf.nn.softmax(logits_fake)[:,1]

            curr_box = end_points['F_pred_box_reg']
            curr_center_reg, curr_size_reg, curr_angle_reg = curr_box
            
            total_delta_center = tf.zeros_like(curr_center_reg)
            total_delta_angle  = tf.zeros_like(curr_angle_reg)
            total_delta_size   = tf.zeros_like(curr_size_reg)
            for i in range(FLAGS.SEMI_REFINE_USING_BOXPC_DELTA_NUM):
                fake_box_pc = (curr_box, pc_pl)
                with tf.variable_scope('D_boxpc_branch', reuse=True):
                    fake_boxpc_pred, fake_boxpc_ep = boxpc_sunrgbd.get_model(fake_box_pc, 
                                                     is_training_D, one_hot_vec_pl, 
                                                     use_one_hot_vec=FLAGS.use_one_hot_boxpc, 
                                                     c=FLAGS)
                
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

            if FLAGS.SEMI_BOXPC_MIN_FIT_LOSS_AFT_REFINE:
                fake_boxpc_fit_prob = tf.nn.softmax(fake_boxpc_ep['boxpc_fit_logits'])[:,1]

            F2_center            = end_points['F_center'] - total_delta_center
            F2_heading_scores    = end_points['F_heading_scores']
            F2_heading_residuals = end_points['F_heading_residuals'] - \
                                   tf_util.tf_expand_tile(total_delta_angle, axis=1, tile=[1,12])
            F2_size_scores       = end_points['F_size_scores']
            F2_size_residuals    = end_points['F_size_residuals'] - \
                                   tf_util.tf_expand_tile(total_delta_size, axis=1, tile=[1,10,1])

            end_points.update({ 'pred_boxpc_fit'       : fake_boxpc_ep['pred_boxpc_fit'],
                                'boxpc_feats_dict'     : fake_boxpc_ep['boxpc_feats_dict'],
                                'boxpc_fit_prob'       : fake_boxpc_fit_prob,
                                'boxpc_delta_center'   : fake_boxpc_ep['boxpc_delta_center'],
                                'boxpc_delta_size'     : fake_boxpc_ep['boxpc_delta_size'],
                                'boxpc_delta_angle'    : fake_boxpc_ep['boxpc_delta_angle'],
                                'F2_center'            : F2_center,
                                'F2_heading_scores'    : F2_heading_scores,
                                'F2_heading_residuals' : F2_heading_residuals,
                                'F2_size_scores'       : F2_size_scores,
                                'F2_size_residuals'    : F2_size_residuals })

            # ======================================= G LOSS =======================================
            semi_loss = MODEL.get_semi_loss(pred, labels, end_points, c=FLAGS)
            train_vars    = get_scope_vars('class_dependent', trainable_only=True)
            if FLAGS.SEMI_TRAIN_BOX_TRAIN_CLASS_AG_TNET:
                train_vars += get_scope_vars('class_agnostic/tnet', trainable_only=True)
            if FLAGS.SEMI_TRAIN_BOX_TRAIN_CLASS_AG_BOX:
                train_vars += get_scope_vars('class_agnostic/box', trainable_only=True)
            train_semi_op = optimizer.minimize(semi_loss, 
                                               global_step=batch,
                                               var_list=train_vars)
            tf.summary.scalar('Total_Loss/semi_loss', semi_loss)
            ops = { 'semi_loss'     : semi_loss,
                    'train_semi_op' : train_semi_op }

            
            correct = tf.equal(tf.argmax(logits, 2), tf.to_int64(y_seg_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('Seg_IOU/accuracy', accuracy)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=5)
        
        # ======================================== LOGS ========================================
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(pjoin(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(pjoin(LOG_DIR, 'test'), sess.graph)

        # ==================================== INIT & RESTORE ====================================
        # Init variables
        if FLAGS.init_class_ag_path is not None:

            # Restore only certain variables
            class_ag_scopes             = ['class_agnostic']
            class_ag_scopenames_in_ckpt = ['']
            load_variable_scopes_from_ckpt(class_ag_scopes, class_ag_scopenames_in_ckpt, 
                                           sess, FLAGS.init_class_ag_path)

            boxpc_scopes             = ['D_boxpc_branch']
            boxpc_scopenames_in_ckpt = ['']
            load_variable_scopes_from_ckpt(boxpc_scopes, boxpc_scopenames_in_ckpt, 
                                           sess, FLAGS.init_boxpc_path[0])

            # Initialize the rest
            already_init_var_scopes = ['class_agnostic', 'D_boxpc_branch']
            init = tf.variables_initializer(get_scope_vars_except_unwanted_scopes(
                                            already_init_var_scopes, trainable_only=False))
            sess.run(init)
                
        elif FLAGS.restore_model_path is None:
            assert(FLAGS.init_class_ag_path is None)
            assert(FLAGS.init_boxpc_path is None)
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

        ops.update({ 'pc_pl'           : pc_pl,
                     'bg_pc_pl'        : bg_pc_pl,
                     'img_pl'          : img_pl,
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
                     'is_data_2D_pl'   : is_data_2D_pl,
                     'is_training_pl'  : is_training_pl,
                     'logits'          : logits,
                     'merged'          : merged,
                     'step'            : batch,
                     'end_points'      : end_points })

        # ====================================== TRAINING ======================================
        best_loss = 1e10
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
                    
            T1.tic()
            train_one_epoch(sess, ops, train_writer)
            epoch_loss = eval_one_epoch(sess, ops, test_writer)
            T1.toc(average=False)

            # Save the variables to disk.
            if epoch % 5 == 0:
                save_path = saver.save(sess, pjoin(LOG_DIR, 'model_epoch_%d.ckpt' % epoch))
                log_string('Model saved in file: %s' % save_path)

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    W_iou2ds_sum = 0
    W_iou3ds_sum = 0
    adv_stats = AdvClassificationStats(ALL_CLASSES)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx   = (batch_idx+1) * BATCH_SIZE

        iters = 2 if FLAGS.SEMI_SAMPLING_METHOD =='ALTERNATE_BATCH' else 1
        for iteration in range(iters):
            # ===================================== SAMPLING DATA =====================================
            # Prepare data generally
            if FLAGS.SEMI_SAMPLING_METHOD == 'BATCH':
                # Iterate through all data
                assert(iteration == 0)
                batch_data, batch_img_crop, batch_label, batch_center, batch_hclass, batch_hres, \
                batch_sclass, batch_sres, batch_box2d, batch_rtilt, batch_k, batch_rot_angle, \
                batch_img_dims, batch_one_hot_vec, batch_is_data_2D = \
                    TRAIN_DATASET.get_batch(train_idxs, start_idx, end_idx, NUM_POINT, FLAGS.NUM_CHANNELS)

            elif FLAGS.SEMI_SAMPLING_METHOD == 'ALTERNATE_BATCH':
                # Alternate between weak and strong samples, 
                # If sample_equal_class_prob = 0: Then randomly sample within the weak samples and within 
                #                                 the strong samples.
                # If sample_equal_class_prob = 1: Sampling ~equal num samples/class within the weak samples 
                #                                 and within the strong samples.
                sample_equal_class = (np.random.rand() < FLAGS.SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB)
                assert(iteration == 0 or iteration == 1)
                sampling_fn = TRAIN_DATASET.sample_pure_from_2D_cls if iteration == 0 else \
                              TRAIN_DATASET.sample_pure_from_3D_cls
                batch_data, batch_img_crop, batch_label, batch_center, batch_hclass, batch_hres, \
                batch_sclass, batch_sres, batch_box2d, batch_rtilt, batch_k, batch_rot_angle, \
                batch_img_dims, batch_one_hot_vec, batch_is_data_2D = \
                    sampling_fn(BATCH_SIZE, NUM_POINT, FLAGS.NUM_CHANNELS,
                                equal_samples_per_class=sample_equal_class)

            elif FLAGS.SEMI_SAMPLING_METHOD == 'MIXED_BATCH':
                # Get weak and strong samples in the same batch, 
                # If sample_equal_class_prob = 0: Then randomly sample within the weak samples and within 
                #                                 the strong samples.
                # If sample_equal_class_prob = 1: Sampling ~equal num samples/class within the weak samples 
                #                                 and within the strong samples.
                sample_equal_class = (np.random.rand() < FLAGS.SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB)
                assert(iteration == 0)
                sampling_fn = TRAIN_DATASET.sample_mixed
                batch_data, batch_img_crop, batch_label, batch_center, batch_hclass, batch_hres, \
                batch_sclass, batch_sres, batch_box2d, batch_rtilt, batch_k, batch_rot_angle, \
                batch_img_dims, batch_one_hot_vec, batch_is_data_2D = \
                    sampling_fn(BATCH_SIZE, NUM_POINT, FLAGS.NUM_CHANNELS,
                                equal_samples_per_class=sample_equal_class)

            # Augment batched point clouds by rotation and jittering
            aug_data = batch_data
            #aug_data = provider.random_scale_point_cloud(batch_data)
            #aug_data = provider.jitter_point_cloud(aug_data)

            # Setup data + run_ops according to training type (strong/semi/weak)
            feed_dict        = { ops['pc_pl']           : aug_data,
                                 #ops['img_pl']          : batch_img_crop,
                                 ops['one_hot_vec_pl']  : batch_one_hot_vec,
                                 ops['is_training_pl']  : is_training,
                                 ops['y_seg_pl']        : batch_label,
                                 ops['y_centers_pl']    : batch_center,
                                 ops['y_orient_cls_pl'] : batch_hclass,
                                 ops['y_orient_reg_pl'] : batch_hres,
                                 ops['y_dims_cls_pl']   : batch_sclass,
                                 ops['y_dims_reg_pl']   : batch_sres,
                                 ops['box2D_pl']        : batch_box2d,
                                 ops['Rtilt_pl']        : batch_rtilt,
                                 ops['K_pl']            : batch_k,
                                 ops['rot_frust_pl']    : np.expand_dims(batch_rot_angle, axis=1),
                                 ops['img_dim_pl']      : batch_img_dims,
                                 ops['is_data_2D_pl']   : batch_is_data_2D }
            run_ops = [ops['merged'], ops['step'], ops['logits']]


            # ======================================= TRAINING =======================================
            iou2ds, iou3ds = 0, 0
            W_iou2ds, W_iou3ds = 0, 0

            run_ops.extend([ops['end_points']['iou2ds'], ops['end_points']['iou3ds'], 
                            ops['semi_loss'], ops['train_semi_op']])
            summary, step, logits_val, iou2ds, iou3ds, loss_val, _ = \
                sess.run(run_ops, feed_dict=feed_dict)
            adv_stats.add_loss(loss_val)
            
            for i in range(FLAGS.SEMI_ADV_ITERS_FOR_D):
                pred_for_real, pred_for_fake, _ = sess.run([ops['end_points']['pred_for_real'], 
                    ops['end_points']['pred_for_fake'], ops['train_D_op']], feed_dict=feed_dict)

                # Stats
                adv_stats.add_prediction((np.squeeze(pred_for_real, axis=1) > 0.5) * 1, 
                                         np.ones(BATCH_SIZE, dtype=int), 
                                         np.argmax(batch_one_hot_vec, axis=1))
                adv_stats.add_prediction((np.squeeze(pred_for_fake, axis=1) > 0.5) * 1, 
                                         np.zeros(BATCH_SIZE, dtype=int), 
                                         np.argmax(batch_one_hot_vec, axis=1))

            # ======================================= Statistics =======================================
            train_writer.add_summary(summary, step)
            preds_val = np.argmax(logits_val, 2)
            correct = np.sum(preds_val == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE*NUM_POINT)
            iou2ds_sum += np.sum(iou2ds)
            iou3ds_sum += np.sum(iou3ds)
            W_iou2ds_sum += np.sum(W_iou2ds)
            W_iou3ds_sum += np.sum(W_iou3ds)

            if (batch_idx + 1) % 1000 == 0:
                log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
                log_string('mean loss: %f' % adv_stats.get_mean_loss())
                log_string('accuracy: %f' % (total_correct / float(total_seen)))
                if iou3ds_sum > 0:
                    log_string('Strong Box IoU (ground/3D): %f / %f' % \
                        (iou2ds_sum / float(BATCH_SIZE*10), iou3ds_sum / float(BATCH_SIZE*10)))
                if W_iou3ds_sum > 0:
                    log_string('Weak Box IoU (ground/3D): %f / %f' % \
                        (W_iou2ds_sum / float(BATCH_SIZE*10), W_iou3ds_sum / float(BATCH_SIZE*10)))
                total_correct = 0
                total_seen = 0
                iou2ds_sum = 0
                iou3ds_sum = 0
                W_iou2ds_sum = 0
                W_iou3ds_sum = 0
                adv_stats.reset_loss()

    if FLAGS.SEMI_ADV_ITERS_FOR_D == 0: return
    batch_stats = adv_stats.get_batch_stats()
    log_string(adv_stats.summarize_stats(batch_stats))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_SEG_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_SEG_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    W_iou2ds_sum = 0
    W_iou3ds_sum = 0
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    # ==================================== AP EVALUATION ====================================
    # Calculate mean AP
    eval_prefix1 = 'F_'
    predictions = main_batch(TEST_DATASET, FLAGS.TEST_CLS, len(ALL_CLASSES), NUM_POINT, 
                             FLAGS.NUM_CHANNELS, 
                             prefix=eval_prefix1, 
                             sess_ops=(sess, ops), 
                             output_filename=None)
    rec, prec, ap, mean_ap = evaluate_predictions(predictions, DATASET, ALL_CLASSES, 
                                                  TEST_DATASET, FLAGS.TEST_CLS)
    add_ap_summary(ap, mean_ap, test_writer, EPOCH_CNT, prefix='AP_Intermediate/')
    log_string(get_ap_info(ap, mean_ap))

    # Final result that we are interested in
    eval_prefix2 = 'F2_'
    predictions = main_batch(TEST_DATASET, FLAGS.TEST_CLS, len(ALL_CLASSES), NUM_POINT, 
                             FLAGS.NUM_CHANNELS, 
                             prefix=eval_prefix2, 
                             sess_ops=(sess, ops), 
                             output_filename=None)
    rec, prec, ap, mean_ap = evaluate_predictions(predictions, DATASET, ALL_CLASSES, 
                                                  TEST_DATASET, FLAGS.TEST_CLS)
    add_ap_summary(ap, mean_ap, test_writer, EPOCH_CNT, prefix='AP_Test/')
    log_string(get_ap_info(ap, mean_ap))

    EPOCH_CNT += 1

class AdvClassificationStats(object):
    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
        self.reset_loss()
        self.reset_stats()

    def add_loss(self, loss):
        self.loss_sum += loss
        self.num_batches += 1

    def add_prediction(self, pred_real, y_real, y_cls):
        self.pred_reals.extend(pred_real)
        self.y_reals.extend(y_real)
        self.y_clses.extend(y_cls)

    def get_mean_loss(self):
        mean_loss = self.loss_sum / self.num_batches
        return mean_loss

    def get_batch_stats(self):
        pred_real = np.array(self.pred_reals)
        y_real = np.array(self.y_reals)
        y_cls = np.array(self.y_clses)
        stats = {}
        for c in range(self.num_classes):
            tp = float( np.sum((y_real == 1) & (pred_real == 1) & (y_cls == c)) )
            fp = float( np.sum((y_real == 0) & (pred_real == 1) & (y_cls == c)) )
            fn = float( np.sum((y_real == 1) & (pred_real == 0) & (y_cls == c)) )
            tn = float( np.sum((y_real == 0) & (pred_real == 0) & (y_cls == c)) )
            prec    = tp / (tp + fp + 1e-3)
            recall  = tp / (tp + fn + 1e-3)
            f1      = (2 * prec * recall) / (prec + recall + 1e-3)
            support = np.sum(y_cls == c)
            if support == 0: continue
            stats[self.classes[c]] = (prec, recall, f1, support)
        return stats

    def summarize_stats(self, stats):
        classes = stats.keys()
        classes.sort()
        summary_string = '%11s %6s %6s %4s %5s\n' % ('  Classname', 'Prec', 'Recall', ' F1 ', 'Supp')
        precs, recalls, f1s, supports = [], [], [], []
        for cls_type in classes:
            prec, recall, f1, support = stats[cls_type]
            precs.append(prec)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)
            summary_string += '%11s: %.3f %.3f %.3f %5d\n' % (cls_type, prec, recall, f1, support)
        summary_string += '%11s: %.3f %.3f %.3f %5d\n' % \
            ('      MEAN ', np.mean(precs), np.mean(recalls), np.mean(f1s), np.mean(supports))
        return summary_string

    def reset_loss(self):
        self.loss_sum = 0
        self.num_batches = 0

    def reset_stats(self):
        self.pred_reals = []
        self.y_reals = []
        self.y_clses = []

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
