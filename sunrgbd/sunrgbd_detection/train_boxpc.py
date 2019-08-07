
""" Train BoxPC model.

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
import boxpc_sunrgbd
import box_pc_fit_dataset
from config import cfg
from timer import Timer
from box_util import box3d_iou
from roi_seg_box3d_dataset import get_3d_box, class2size, class2angle

cfg.add_argument('--train_data', type=str, required=True, choices=['train_mini', 'train_aug5x', 'trainval_aug5x'])
cfg.add_argument('--classes_to_drop_prob', type=float, default=1, help='Percentage of 3D data to drop randomly. 1 means all data is dropped.')
cfg.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
cfg.add_argument('--log_dir', default='log', help='Log dir [default: log]')
cfg.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
cfg.add_argument('--max_epoch', type=int, default=31, help='Epoch to run [default: 51]')
cfg.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
cfg.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
cfg.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
cfg.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
cfg.add_argument('--decay_step', type=int, default=800000, help='Decay step for lr decay [default: 200000]')
cfg.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
cfg.add_argument('--use_mini', action='store_true', help='Use mini dataset for training data')
cfg.add_argument('--train_all', action='store_true', help='Train on all 10 classes')
cfg.add_argument('--use_one_hot', action='store_true', help='Use one hot vector during training')
cfg.add_argument('--no_rgb', action='store_true', help='Only use XYZ for training')
cfg.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
FLAGS = cfg.parse_special_args()

# Parameters
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
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
if FLAGS.train_all:
    FLAGS.TRAIN_CLS = ALL_CLASSES
    FLAGS.TEST_CLS  = ALL_CLASSES


# Files and logs
TRAIN_FILE   = pjoin(BASE_DIR, 'train_boxpc.py')
MODEL_FILE   = pjoin(BASE_DIR, 'boxpc_sunrgbd.py')
MODELS_FILE  = pjoin(BASE_DIR, 'semisup_models.py')
WLOSSES_FILE = pjoin(MODEL_DIR, 'weak_losses.py')
CONFIG_FILE  = pjoin(MODEL_DIR, 'config.py')
LOG_DIR      = pjoin('experiments_boxpc', FLAGS.log_dir)
if not os.path.exists('experiments_boxpc'): os.mkdir('experiments_boxpc')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# Backup of relevant code
for file in [TRAIN_FILE, MODEL_FILE, MODELS_FILE, WLOSSES_FILE, CONFIG_FILE]:
    os.system('cp %s %s' % (file, LOG_DIR))
LOG_FOUT = open(pjoin(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
T1 = Timer('1 Epoch')
T2 = Timer('Preparing data for 1 Epoch')
T3 = Timer('Preparing testing data for 1 Epoch')


# Datasets
train_file = 'train_mini.zip.pickle' if FLAGS.use_mini else ('%s.zip.pickle' % FLAGS.train_data)
test_file  = 'train_mini.zip.pickle' if FLAGS.use_mini else 'test.zip.pickle'
TRAIN_DATASET = box_pc_fit_dataset.BoxPCFitDataset(
                classes=ALL_CLASSES,
                classes_to_drop=FLAGS.TEST_CLS,
                classes_to_drop_prob=FLAGS.classes_to_drop_prob,
                center_perturbation=FLAGS.BOXPC_CENTER_PERTURBATION,
                size_perturbation=FLAGS.BOXPC_SIZE_PERTURBATION,
                angle_perturbation=FLAGS.BOXPC_ANGLE_PERTURBATION,
                npoints=NUM_POINT, rotate_to_center=True, random_flip=True, random_shift=True, 
                overwritten_data_path=pjoin('frustums',train_file))
print('Length of Train Dataset: %d' % len(TRAIN_DATASET))

np.random.seed(1)
TEST_DATASET = box_pc_fit_dataset.BoxPCFitDataset(classes=ALL_CLASSES, 
               center_perturbation=FLAGS.BOXPC_CENTER_PERTURBATION,
               size_perturbation=FLAGS.BOXPC_SIZE_PERTURBATION,
               angle_perturbation=FLAGS.BOXPC_ANGLE_PERTURBATION,
               npoints=NUM_POINT, rotate_to_center=True, random_flip=True, random_shift=True, 
               overwritten_data_path=pjoin('frustums',test_file))
T3.tic()
TEST_DATASET.prepare_batches_for_one_epoch('BATCH', 
             len(TEST_DATASET)/BATCH_SIZE, BATCH_SIZE, NUM_POINT, FLAGS.NUM_CHANNELS, 
             equal_classes_prob=None,
             proportion_of_boxpc_fit=0.5,
             boxpc_nofit_bounds=FLAGS.BOXPC_NOFIT_BOUNDS,
             boxpc_fit_bounds=FLAGS.BOXPC_FIT_BOUNDS)
print('Length of Test Dataset : %d' % len(TEST_DATASET))
T3.toc(average=False)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
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

def keep_only_ids(items, ids):
    return [item[ids,...] for item in items]

def add_class_stats_summary(cls_stats, writer, step):
    classes = cls_stats.keys()
    classes.sort()
    precs, recalls, f1s, supports = [], [], [], []
    for cls_type in classes:
        prec, recall, f1, support = cls_stats[cls_type]
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

def add_box_delta_stats_summary(box_stats, writer, step, prefix=''):
    classes = box_stats.keys()
    classes.sort()
    mean_iou3d_oris, mean_iou3d_dels, change_in_iou3ds, supports = [], [], [], []
    for cls_type in classes:
        mean_iou3d_ori, mean_iou3d_del, change_in_iou3d, support = box_stats[cls_type]
        mean_iou3d_oris.append(mean_iou3d_ori)
        mean_iou3d_dels.append(mean_iou3d_del)
        change_in_iou3ds.append(change_in_iou3d)
        supports.append(support)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Box_IOU_Before/%s%s' % (cls_type, prefix), simple_value=mean_iou3d_ori)]), step)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Box_IOU_After/%s%s' % (cls_type, prefix), simple_value=mean_iou3d_del)]), step)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Box_IOU_Change/%s%s' % (cls_type, prefix), simple_value=change_in_iou3d)]), step)
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Box_Support/%s%s' % (cls_type, prefix), simple_value=support)]), step)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Box_IOU_Before/Mean%s' % (prefix), simple_value=np.mean(mean_iou3d_oris))]), step)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Box_IOU_After/Mean%s' % (prefix), simple_value=np.mean(mean_iou3d_dels))]), step)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Box_IOU_Change/Mean%s' % (prefix), simple_value=np.mean(change_in_iou3ds))]), step)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Box_Support/Mean%s' % (prefix), simple_value=np.mean(supports))]), step)

##############################################################################
# Training
##############################################################################

def train():
    # Print configurations
    log_string('\n\nCommand:\npython %s\n' % ' '.join(sys.argv))
    log_string(FLAGS.config_str)

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pc_pl, one_hot_vec_pl, y_seg_pl, x_center_pl, x_orient_cls_pl, x_orient_reg_pl, \
            x_dims_cls_pl, x_dims_reg_pl, y_box_iou_pl, y_center_delta_pl, y_dims_delta_pl, \
            y_orient_delta_pl, = \
                boxpc_sunrgbd.placeholder_inputs(BATCH_SIZE, NUM_POINT, FLAGS.NUM_CHANNELS)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            x_box = (x_center_pl, x_orient_cls_pl, x_orient_reg_pl, x_dims_cls_pl, x_dims_reg_pl)
            box_reg = boxpc_sunrgbd.convert_raw_y_box_to_reg_format(x_box, one_hot_vec_pl)
            boxpc = (box_reg, pc_pl)
            labels = (y_box_iou_pl, (y_center_delta_pl, y_dims_delta_pl, y_orient_delta_pl))
            
            # Note the global_step=batch parameter to minimize. That tells the optimizer to 
            # helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('Info/bn_decay', bn_decay)
            
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('Info/learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer   = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer   = tf.train.AdamOptimizer(learning_rate)

            # Get model and loss
            pred, end_points = boxpc_sunrgbd.get_model(boxpc, is_training_pl, one_hot_vec_pl, 
                                                       use_one_hot_vec=FLAGS.use_one_hot,
                                                       bn_decay=bn_decay, c=FLAGS)
            loss     = boxpc_sunrgbd.get_loss(pred, labels, end_points, c=FLAGS)
            train_op = optimizer.minimize(loss, global_step=batch)
            tf.summary.scalar('Total_Loss/loss', loss)
            pred_boxpc_fit = end_points['pred_boxpc_fit']

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=5)
        
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

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

        ops = { 'pc_pl'             : pc_pl,
                'one_hot_vec_pl'    : one_hot_vec_pl,
                'x_center_pl'       : x_center_pl,
                'x_orient_cls_pl'   : x_orient_cls_pl,
                'x_orient_reg_pl'   : x_orient_reg_pl,
                'x_dims_cls_pl'     : x_dims_cls_pl,
                'x_dims_reg_pl'     : x_dims_reg_pl,
                'y_box_iou_pl'      : y_box_iou_pl,
                'y_center_delta_pl' : y_center_delta_pl,
                'y_orient_delta_pl' : y_orient_delta_pl,
                'y_dims_delta_pl'   : y_dims_delta_pl,
                'is_training_pl'    : is_training_pl,
                'pred_boxpc_fit'    : pred_boxpc_fit,
                'loss'              : loss,
                'train_op'          : train_op,
                'merged'            : merged,
                'step'              : batch,
                'end_points'        : end_points }

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
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE

    # Prepare data beforehand
    T2.tic()
    TRAIN_DATASET.prepare_batches_for_one_epoch(FLAGS.BOXPC_SAMPLING_METHOD, 
                  num_batches, BATCH_SIZE, NUM_POINT, FLAGS.NUM_CHANNELS, 
                  equal_classes_prob=FLAGS.BOXPC_SAMPLE_EQUAL_CLASS_WITH_PROB,
                  proportion_of_boxpc_fit=FLAGS.BOXPC_PROPORTION_OF_BOXPC_FIT,
                  boxpc_nofit_bounds=FLAGS.BOXPC_NOFIT_BOUNDS,
                  boxpc_fit_bounds=FLAGS.BOXPC_FIT_BOUNDS)
    T2.toc(average=False)
    
    log_string(str(datetime.now()))

    cls_stats = ClassificationStats(ALL_CLASSES)
    box_stats_for_pos = BoxDeltaIOUStats(ALL_CLASSES)
    box_stats_for_neg = BoxDeltaIOUStats(ALL_CLASSES)
    for batch_idx in range(num_batches):

        batch_data, batch_image, batch_label, batch_center, batch_heading_class, \
        batch_heading_residual, batch_size_class, batch_size_residual, batch_box2d, \
        batch_rilts, batch_ks, batch_rot_angle, batch_img_dims, batch_one_hot_vec, \
        batch_new_center, batch_new_heading_class, batch_new_heading_residual, \
        batch_new_size_class, batch_new_size_residual, batch_box_iou, batch_y_center_delta, \
        batch_y_size_delta, batch_y_angle_delta = TRAIN_DATASET.get_prepared_batch()

        # Setup data + run_ops
        feed_dict = { ops['pc_pl']             : batch_data,
                      ops['one_hot_vec_pl']    : batch_one_hot_vec,
                      ops['x_center_pl']       : batch_new_center,
                      ops['x_orient_cls_pl']   : batch_new_heading_class,
                      ops['x_orient_reg_pl']   : batch_new_heading_residual,
                      ops['x_dims_cls_pl']     : batch_new_size_class,
                      ops['x_dims_reg_pl']     : batch_new_size_residual,
                      ops['y_box_iou_pl']      : batch_box_iou,
                      ops['y_center_delta_pl'] : batch_y_center_delta,
                      ops['y_dims_delta_pl']   : batch_y_size_delta,
                      ops['y_orient_delta_pl'] : batch_y_angle_delta, 
                      ops['is_training_pl']    : is_training }
    
        ep = ops['end_points']
        run_ops = [ops['merged'], ops['step'], ops['loss'], ops['train_op'], 
                   ops['pred_boxpc_fit'], 
                   ep['boxpc_delta_center'], 
                   ep['boxpc_delta_size'], 
                   ep['boxpc_delta_angle']]
        summary, step, loss_val, _, pred_boxpc_fit, del_center, del_size_residual, del_heading_residual = \
            sess.run(run_ops, feed_dict=feed_dict)

        # Statistics
        train_writer.add_summary(summary, step)
        y_cls = np.argmax(batch_one_hot_vec, axis=1)
        cls_stats.add_prediction(pred_boxpc_fit, (batch_box_iou > FLAGS.BOXPC_FIT_BOUNDS[0]) * 1, y_cls)
        cls_stats.add_loss(loss_val)

        y_box = [batch_center, batch_heading_class, batch_heading_residual, 
                 batch_size_class, batch_size_residual]
        ori_box = [batch_new_center, batch_new_heading_class, 
                   batch_new_heading_residual, batch_new_size_class, 
                   batch_new_size_residual]
        del_box = [batch_new_center - del_center, 
                   batch_new_heading_class, 
                   batch_new_heading_residual - del_heading_residual, 
                   batch_new_size_class, 
                   batch_new_size_residual - del_size_residual]
        pos_samples = (batch_box_iou >= FLAGS.BOXPC_FIT_BOUNDS[0])
        neg_samples = (batch_box_iou < FLAGS.BOXPC_FIT_BOUNDS[0])
        ori_box_pos = keep_only_ids(ori_box, pos_samples)
        ori_box_neg = keep_only_ids(ori_box, neg_samples)
        del_box_pos = keep_only_ids(del_box, pos_samples)
        del_box_neg = keep_only_ids(del_box, neg_samples)
        y_box_pos = keep_only_ids(y_box, pos_samples)
        y_box_neg = keep_only_ids(y_box, neg_samples)
        y_cls_pos = y_cls[pos_samples,...]
        y_cls_neg = y_cls[neg_samples,...]

        box_stats_for_pos.add_prediction(ori_box_pos, del_box_pos, y_box_pos, y_cls_pos)
        box_stats_for_neg.add_prediction(ori_box_neg, del_box_neg, y_box_neg, y_cls_neg)

        if (batch_idx + 1) % 1000 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % cls_stats.get_mean_loss())
            cls_stats.reset_loss()

    batch_stats = cls_stats.get_batch_stats()
    log_string('CLASSIFICATION:\n' + cls_stats.summarize_stats(batch_stats))

    batch_box_stats_pos = box_stats_for_pos.get_batch_stats()
    batch_box_stats_neg = box_stats_for_neg.get_batch_stats()
    log_string('POSITIVE EXAMPLES:\n' + box_stats_for_pos.summarize_stats(batch_box_stats_pos))
    log_string('NEGATIVE EXAMPLES:\n' + box_stats_for_neg.summarize_stats(batch_box_stats_neg))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    cls_stats = ClassificationStats(ALL_CLASSES)
    box_stats_for_pos = BoxDeltaIOUStats(ALL_CLASSES)
    box_stats_for_neg = BoxDeltaIOUStats(ALL_CLASSES)
    for batch_idx in range(num_batches):

        # Prepare data generally
        batch_data, batch_image, batch_label, batch_center, batch_heading_class, \
        batch_heading_residual, batch_size_class, batch_size_residual, batch_box2d, \
        batch_rilts, batch_ks, batch_rot_angle, batch_img_dims, batch_one_hot_vec, \
        batch_new_center, batch_new_heading_class, batch_new_heading_residual, \
        batch_new_size_class, batch_new_size_residual, batch_box_iou, batch_y_center_delta, \
        batch_y_size_delta, batch_y_angle_delta = \
            TEST_DATASET.get_prepared_batch_without_removing(batch_idx)

        # Setup data + run_ops
        feed_dict = { ops['pc_pl']             : batch_data,
                      ops['one_hot_vec_pl']    : batch_one_hot_vec,
                      ops['x_center_pl']       : batch_new_center,
                      ops['x_orient_cls_pl']   : batch_new_heading_class,
                      ops['x_orient_reg_pl']   : batch_new_heading_residual,
                      ops['x_dims_cls_pl']     : batch_new_size_class,
                      ops['x_dims_reg_pl']     : batch_new_size_residual,
                      ops['y_box_iou_pl']      : batch_box_iou,
                      ops['y_center_delta_pl'] : batch_y_center_delta,
                      ops['y_dims_delta_pl']   : batch_y_size_delta,
                      ops['y_orient_delta_pl'] : batch_y_angle_delta, 
                      ops['is_training_pl']    : is_training }
        
        ep = ops['end_points']
        run_ops = [ops['merged'], ops['step'], ops['loss'], 
                   ops['pred_boxpc_fit'],
                   ep['boxpc_delta_center'], 
                   ep['boxpc_delta_size'], 
                   ep['boxpc_delta_angle']]
        summary, step, loss_val, pred_boxpc_fit, del_center, del_size_residual, del_heading_residual = \
            sess.run(run_ops, feed_dict=feed_dict)

        # Statistics
        test_writer.add_summary(summary, step)
        y_cls = np.argmax(batch_one_hot_vec, axis=1)
        cls_stats.add_prediction(pred_boxpc_fit, (batch_box_iou > FLAGS.BOXPC_FIT_BOUNDS[0]) * 1, y_cls)
        cls_stats.add_loss(loss_val)

        y_box = [batch_center, batch_heading_class, batch_heading_residual, 
                 batch_size_class, batch_size_residual]
        ori_box = [batch_new_center, batch_new_heading_class, 
                   batch_new_heading_residual, batch_new_size_class, 
                   batch_new_size_residual]
        del_box = [batch_new_center - del_center, 
                   batch_new_heading_class, 
                   batch_new_heading_residual - del_heading_residual, 
                   batch_new_size_class, 
                   batch_new_size_residual - del_size_residual]
        pos_samples = (batch_box_iou >= FLAGS.BOXPC_FIT_BOUNDS[0])
        neg_samples = (batch_box_iou < FLAGS.BOXPC_FIT_BOUNDS[0])
        ori_box_pos = keep_only_ids(ori_box, pos_samples)
        ori_box_neg = keep_only_ids(ori_box, neg_samples)
        del_box_pos = keep_only_ids(del_box, pos_samples)
        del_box_neg = keep_only_ids(del_box, neg_samples)
        y_box_pos = keep_only_ids(y_box, pos_samples)
        y_box_neg = keep_only_ids(y_box, neg_samples)
        y_cls_pos = y_cls[pos_samples,...]
        y_cls_neg = y_cls[neg_samples,...]

        box_stats_for_pos.add_prediction(ori_box_pos, del_box_pos, y_box_pos, y_cls_pos)
        box_stats_for_neg.add_prediction(ori_box_neg, del_box_neg, y_box_neg, y_cls_neg)

    batch_stats = cls_stats.get_batch_stats()
    add_class_stats_summary(batch_stats, test_writer, EPOCH_CNT)
    log_string('CLASSIFICATION:\n' + cls_stats.summarize_stats(batch_stats))
    log_string('eval mean loss: %f' % cls_stats.get_mean_loss())

    batch_box_stats_pos = box_stats_for_pos.get_batch_stats()
    batch_box_stats_neg = box_stats_for_neg.get_batch_stats()
    add_box_delta_stats_summary(batch_box_stats_pos, test_writer, EPOCH_CNT, prefix='_POS')
    add_box_delta_stats_summary(batch_box_stats_neg, test_writer, EPOCH_CNT, prefix='_NEG')
    log_string('POSITIVE EXAMPLES:\n' + box_stats_for_pos.summarize_stats(batch_box_stats_pos))
    log_string('NEGATIVE EXAMPLES:\n' + box_stats_for_neg.summarize_stats(batch_box_stats_neg))

    EPOCH_CNT += 1
    return cls_stats.get_mean_loss()

##############################################################################
# Performance Statistics
##############################################################################

class BoxDeltaIOUStats(object):
    """
    Evaluates the box delta terms predicted by the BoxPC model by measuring
    hwo much it improves the IOU of a given box wrt the ground-truth.
    """
    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
        self.reset_stats()

    def add_prediction(self, ori_box, ori_box_aft_delta, y_box, y_cls):
        """
        ori_box: original box that has been fed to the Box-PC network for cls + reg
        ori_box_aft_delta: delta term applied on center, size, angle of ori_box
        y_box: ground truth box
        y_cls: classes of the data points
        """
        ori_center, ori_heading_class, ori_heading_residual, ori_size_class, ori_size_residual = ori_box
        del_center, del_heading_class, del_heading_residual, del_size_class, del_size_residual = ori_box_aft_delta
        y_center, y_heading_class, y_heading_residual, y_size_class, y_size_residual = y_box

        self.ori_centers.extend(ori_center)
        self.ori_heading_classes.extend(ori_heading_class)
        self.ori_heading_residuals.extend(ori_heading_residual)
        self.ori_size_classes.extend(ori_size_class)
        self.ori_size_residuals.extend(ori_size_residual)
        
        self.del_centers.extend(del_center)
        self.del_heading_classes.extend(del_heading_class)
        self.del_heading_residuals.extend(del_heading_residual)
        self.del_size_classes.extend(del_size_class)
        self.del_size_residuals.extend(del_size_residual)
        
        self.y_centers.extend(y_center)
        self.y_heading_classes.extend(y_heading_class)
        self.y_heading_residuals.extend(y_heading_residual)
        self.y_size_classes.extend(y_size_class)
        self.y_size_residuals.extend(y_size_residual)

        self.y_clses.extend(y_cls)

    def get_batch_stats(self):
        ori_centers = np.array(self.ori_centers)
        ori_heading_classes = np.array(self.ori_heading_classes)
        ori_heading_residuals = np.array(self.ori_heading_residuals)
        ori_size_classes = np.array(self.ori_size_classes)
        ori_size_residuals = np.array(self.ori_size_residuals)
        
        del_centers = np.array(self.del_centers)
        del_heading_classes = np.array(self.del_heading_classes)
        del_heading_residuals = np.array(self.del_heading_residuals)
        del_size_classes = np.array(self.del_size_classes)
        del_size_residuals = np.array(self.del_size_residuals)
        
        y_centers = np.array(self.y_centers)
        y_heading_classes = np.array(self.y_heading_classes)
        y_heading_residuals = np.array(self.y_heading_residuals)
        y_size_classes = np.array(self.y_size_classes)
        y_size_residuals = np.array(self.y_size_residuals)

        y_clses = np.array(self.y_clses)
        
        assert(ori_centers.shape == del_centers.shape == y_centers.shape)
        assert(ori_size_classes.shape == del_size_classes.shape)
        assert(len(ori_centers) == len(del_centers) == len(y_centers) == len(y_clses))

        iou3ds_ori, iou3ds_del = [], []
        for i in range(len(ori_centers)):
            ori_lwh = class2size(ori_size_classes[i], ori_size_residuals[i])
            ori_ry  = class2angle(ori_heading_classes[i], ori_heading_residuals[i], 12)
            ori_box3d = get_3d_box(ori_lwh, ori_ry, ori_centers[i])

            del_lwh = class2size(del_size_classes[i], del_size_residuals[i])
            del_ry  = class2angle(del_heading_classes[i], del_heading_residuals[i], 12)
            del_box3d = get_3d_box(del_lwh, del_ry, del_centers[i])

            y_lwh = class2size(y_size_classes[i], y_size_residuals[i])
            y_ry  = class2angle(y_heading_classes[i], y_heading_residuals[i], 12)
            y_box3d = get_3d_box(y_lwh, y_ry, y_centers[i])

            ori_iou3d, _ = box3d_iou(y_box3d, ori_box3d)
            del_iou3d, _ = box3d_iou(y_box3d, del_box3d)
            iou3ds_ori.append(ori_iou3d)
            iou3ds_del.append(del_iou3d)
        iou3ds_ori = np.array(iou3ds_ori)
        iou3ds_del = np.array(iou3ds_del)

        stats = {}
        for c in range(self.num_classes):
            is_curr_class = (y_clses == c)
            support = np.sum(is_curr_class)
            if support == 0: continue
            iou3ds_ori_curr_class = iou3ds_ori[is_curr_class,...]
            iou3ds_del_curr_class = iou3ds_del[is_curr_class,...]
            mean_iou3d_ori = np.mean(iou3ds_ori_curr_class)
            mean_iou3d_del = np.mean(iou3ds_del_curr_class)
            change_in_iou3d = mean_iou3d_del - mean_iou3d_ori
            stats[self.classes[c]] = (mean_iou3d_ori, mean_iou3d_del, change_in_iou3d, support)
        return stats

    def summarize_stats(self, stats):
        classes = stats.keys()
        classes.sort()
        summary_string = '%11s %6s %6s %6s %5s\n' % ('  Classname', 'Before', 'After ', ' +/- ', 'Supp')
        mean_iou3d_oris, mean_iou3d_dels, change_in_iou3ds, supports = [], [], [], []
        for cls_type in classes:
            mean_iou3d_ori, mean_iou3d_del, change_in_iou3d, support = stats[cls_type]
            change_in_iou3d_str = ' +' if change_in_iou3d > 0 else ' '
            change_in_iou3d_str += '%.3f' % change_in_iou3d
            mean_iou3d_oris.append(mean_iou3d_ori)
            mean_iou3d_dels.append(mean_iou3d_del)
            change_in_iou3ds.append(change_in_iou3d)
            supports.append(support)
            summary_string += '%11s: %.3f %.3f %4s %5d\n' % \
                (cls_type, mean_iou3d_ori, mean_iou3d_del, change_in_iou3d_str, support)

        change_in_mean_iou3d = np.mean(change_in_iou3ds)
        change_in_mean_iou3d_str = ' +' if change_in_mean_iou3d > 0 else ' '
        change_in_mean_iou3d_str += '%.3f' % change_in_mean_iou3d
        summary_string += '%11s: %.3f %.3f %4s %5d\n' % ('      MEAN ', np.mean(mean_iou3d_oris), \
            np.mean(mean_iou3d_dels), change_in_mean_iou3d_str, np.mean(supports))
        return summary_string

    def reset_stats(self):
        self.ori_centers = []
        self.ori_heading_classes = []
        self.ori_heading_residuals = []
        self.ori_size_classes = []
        self.ori_size_residuals = []
        
        self.del_centers = []
        self.del_heading_classes = []
        self.del_heading_residuals = []
        self.del_size_classes = []
        self.del_size_residuals = []
        
        self.y_centers = []
        self.y_heading_classes = []
        self.y_heading_residuals = []
        self.y_size_classes = []
        self.y_size_residuals = []

        self.y_clses = []

class ClassificationStats(object):
    """
    Evaluates classification predictions of a model.
    """
    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
        self.reset_loss()
        self.reset_stats()

    def add_loss(self, loss):
        self.loss_sum += loss
        self.num_batches += 1

    def add_prediction(self, pred_fit, y_fit, y_cls):
        self.pred_fits.extend(pred_fit)
        self.y_fits.extend(y_fit)
        self.y_clses.extend(y_cls)

    def get_mean_loss(self):
        mean_loss = self.loss_sum / self.num_batches
        return mean_loss

    def get_batch_stats(self):
        pred_fit = np.array(self.pred_fits)
        y_fit = np.array(self.y_fits)
        y_cls = np.array(self.y_clses)
        assert(pred_fit.shape == y_fit.shape == y_cls.shape)
        stats = {}
        for c in range(self.num_classes):
            support = np.sum(y_cls == c)
            if support == 0: continue
            tp = float(np.sum((y_fit == 1) & (pred_fit == 1) & (y_cls == c)))
            fp = float(np.sum((y_fit == 0) & (pred_fit == 1) & (y_cls == c)))
            fn = float(np.sum((y_fit == 1) & (pred_fit == 0) & (y_cls == c)))
            tn = float(np.sum((y_fit == 0) & (pred_fit == 0) & (y_cls == c)))
            prec    = tp / (tp + fp + 1e-3)
            recall  = tp / (tp + fn + 1e-3)
            f1      = (2 * prec * recall) / (prec + recall + 1e-3)
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
        self.pred_fits = []
        self.y_fits = []
        self.y_clses = []

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
