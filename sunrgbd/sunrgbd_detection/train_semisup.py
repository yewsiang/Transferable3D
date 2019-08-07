
""" Train Backbone of Semi-supervised model.

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
cfg.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
cfg.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
cfg.add_argument('--decay_step', type=int, default=800000, help='Decay step for lr decay [default: 200000]')
cfg.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
cfg.add_argument('--use_mini', action='store_true', help='Use mini dataset for training data')
cfg.add_argument('--use_one_hot', action='store_true', help='Use one hot vector during training')
cfg.add_argument('--no_aug', action='store_true', help='Do not augment data during training')
cfg.add_argument('--no_rgb', action='store_true', help='Only use XYZ for training')
cfg.add_argument('--init_model_path', default=None, help='Model parameters for class agnostic branch e.g. log/model.ckpt [default: None]')
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


# Files and logs
MODEL        = importlib.import_module(FLAGS.model) # import network module
TRAIN_FILE   = pjoin(BASE_DIR, 'train_semisup.py')
MODEL_FILE   = pjoin(BASE_DIR, FLAGS.model + '.py')
MODELS_FILE  = pjoin(BASE_DIR, 'semisup_models.py')
WLOSSES_FILE = pjoin(MODEL_DIR, 'weak_losses.py')
CONFIG_FILE  = pjoin(MODEL_DIR, 'config.py')
LOG_DIR      = pjoin('experiments', FLAGS.log_dir)
if not os.path.exists('experiments'): os.mkdir('experiments')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# Backup of relevant code
for file in [TRAIN_FILE, MODEL_FILE, MODELS_FILE, WLOSSES_FILE, CONFIG_FILE]:
    os.system('cp %s %s' % (file, LOG_DIR))
LOG_FOUT = open(pjoin(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
T1 = Timer('1 Epoch')


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


# For Evaluation
from sunrgbd_data import sunrgbd_object
from test_semisup import main_batch, main_batch_from_rgb_detection
from evaluate import evaluate_predictions, get_ap_info

SUNRGBD_DATASET_DIR = '/home/yewsiang/Transferable3D/dataset/mysunrgbd'
DATASET = sunrgbd_object(SUNRGBD_DATASET_DIR, 'training')


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

def get_scope_vars_except(unwanted_scope, trainable_only=False):
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

def add_ap_summary(ap, mean_ap, writer, step, prefix='AP/'):
    for i, classname in enumerate(ap.keys()):
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='%s%s' % (prefix, classname), simple_value=ap[classname])]), step)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='%sMean' % prefix, simple_value=mean_ap)]), step)

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
            
            # Note the global_step=batch parameter to minimize. That tells the optimizer to 
            # helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('Info/bn_decay', bn_decay)
            batch_D = tf.Variable(0)
            bn_decay_D = get_bn_decay(batch_D)
            tf.summary.scalar('Info/bn_decay_D', bn_decay_D)
            
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('Info/learning_rate', learning_rate)
            learning_rate_D = get_learning_rate(batch_D)
            tf.summary.scalar('Info/learning_rate_D', learning_rate_D)
            if OPTIMIZER == 'momentum':
                optimizer   = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                optimizer_D = tf.train.MomentumOptimizer(learning_rate_D, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer   = tf.train.AdamOptimizer(learning_rate)
                optimizer_D = tf.train.AdamOptimizer(learning_rate_D)

            # Get model and loss
            labels = (y_seg_pl, y_centers_pl, y_orient_cls_pl, y_orient_reg_pl, y_dims_cls_pl, \
                      y_dims_reg_pl, R0_rect_pl, P_pl, Rtilt_pl, K_pl, rot_frust_pl, box2D_pl, \
                      img_dim_pl, is_data_2D_pl)
            
            # NOTE: Only use ONE optimizer during each training to prevent batch + optimizer values 
            # getting affected
            norm_box2D = tf_util.tf_normalize_2D_bboxes(box2D_pl, img_dim_pl)
            
            ops = {}
            pred, end_points = MODEL.get_semi_model(pc_pl, bg_pc_pl, img_pl, one_hot_vec_pl, 
                                     is_training_pl, use_one_hot=FLAGS.use_one_hot, 
                                     norm_box2D=norm_box2D, bn_decay=bn_decay, c=FLAGS)
            logits = pred[0]

            semi_loss     = MODEL.get_semi_loss(pred, labels, end_points, c=FLAGS)
            train_semi_op = optimizer.minimize(semi_loss, global_step=batch)
            tf.summary.scalar('Total_Loss/semi_loss', semi_loss)
            ops.update({ 'semi_loss'       : semi_loss,
                         'train_semi_op'   : train_semi_op })
            
            correct = tf.equal(tf.argmax(logits, 2), tf.to_int64(y_seg_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('Seg_IOU/accuracy', accuracy)

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
            assert(FLAGS.init_model_path is None)
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

def train_one_epoch(sess, ops, train_writer, adversary_only=False):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE
    
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx   = (batch_idx+1) * BATCH_SIZE

        iters = 2 if FLAGS.SEMI_SAMPLING_METHOD =='ALTERNATE_BATCH' else 1
        for iteration in range(iters):
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
            run_ops = [ops['merged'], ops['step'], ops['logits'], 
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds'], 
                ops['semi_loss'], ops['train_semi_op']]

            iou2ds, iou3ds = 0, 0
            summary, step, logits_val, iou2ds, iou3ds, loss_val, _ = \
                sess.run(run_ops, feed_dict=feed_dict)

            # Statistics
            train_writer.add_summary(summary, step)
            preds_val = np.argmax(logits_val, 2)
            correct = np.sum(preds_val == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE*NUM_POINT)
            loss_sum += loss_val
            iou2ds_sum += np.sum(iou2ds)
            iou3ds_sum += np.sum(iou3ds)

            if (batch_idx + 1) % 1000 == 0:
                log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
                log_string('mean loss: %f' % (loss_sum / 10))
                log_string('accuracy: %f' % (total_correct / float(total_seen)))
                if iou3ds_sum > 0:
                    log_string('Strong Box IoU (ground/3D): %f / %f' % \
                        (iou2ds_sum / float(BATCH_SIZE*10), iou3ds_sum / float(BATCH_SIZE*10)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                iou2ds_sum = 0
                iou3ds_sum = 0

        
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
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    # Calculate mean AP
    # Final result that we are interested in
    predictions = main_batch(TEST_DATASET, FLAGS.TEST_CLS, len(ALL_CLASSES), NUM_POINT, 
                             FLAGS.NUM_CHANNELS, 
                             prefix='', 
                             sess_ops=(sess, ops), 
                             output_filename=None)
    rec, prec, ap, mean_ap = evaluate_predictions(predictions, DATASET, ALL_CLASSES, 
                                                  TEST_DATASET, FLAGS.TEST_CLS)
    add_ap_summary(ap, mean_ap, test_writer, EPOCH_CNT, prefix='AP_Test/')
    log_string(get_ap_info(ap, mean_ap))
    
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx   = (batch_idx+1) * BATCH_SIZE

        # Prepare data generally
        batch_data, batch_img_crop, batch_label, batch_center, batch_hclass, batch_hres, \
        batch_sclass, batch_sres, batch_box2d, batch_rtilt, batch_k, batch_rot_angle, \
        batch_img_dims, batch_one_hot_vec = \
            TEST_DATASET.get_batch(test_idxs, start_idx, end_idx, NUM_POINT, FLAGS.NUM_CHANNELS)
        batch_is_data_2D = np.zeros(BATCH_SIZE)

        # Setup data + run_ops according to training type (strong/semi/weak)
        feed_dict = { ops['pc_pl']           : batch_data,
                      #ops['img_pl']          : batch_img_crop,
                      ops['one_hot_vec_pl']  : batch_one_hot_vec,
                      ops['y_seg_pl']        : batch_label,
                      ops['y_centers_pl']    : batch_center,
                      ops['y_orient_cls_pl'] : batch_hclass,
                      ops['y_orient_reg_pl'] : batch_hres,
                      ops['y_dims_cls_pl']   : batch_sclass,
                      ops['y_dims_reg_pl']   : batch_sres,
                      ops['is_training_pl']  : is_training,
                      ops['box2D_pl']        : batch_box2d,
                      ops['Rtilt_pl']        : batch_rtilt,
                      ops['K_pl']            : batch_k,
                      ops['rot_frust_pl']    : np.expand_dims(batch_rot_angle, axis=1),
                      ops['img_dim_pl']      : batch_img_dims,
                      ops['is_data_2D_pl']   : batch_is_data_2D,
                      ops['y_seg_pl']        : batch_label }
        run_ops = [ops['merged'], ops['step'], ops['logits'], ops['end_points']['iou2ds'], 
                   ops['end_points']['iou3ds'], ops['semi_loss']]
        summary, step, logits_val, iou2ds, iou3ds, loss_val = \
            sess.run(run_ops, feed_dict=feed_dict)

        # Statistics
        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        for l in range(NUM_SEG_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)
            total_correct_class[l] += (np.sum((preds_val==l) & (batch_label==l)))
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)

        for i in range(BATCH_SIZE):
            segp = preds_val[i,:]
            segl = batch_label[i,:] 
            part_ious = [0.0 for _ in range(NUM_SEG_CLASSES)]
            for l in range(NUM_SEG_CLASSES):
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): 
                    # part is not present, no logitsiction as well
                    part_ious[l] = 1.0
                else:
                    part_ious[l] = np.sum((segl==l) & (segp==l)) / float(np.sum((segl==l) | (segp==l)))
            shape_ious.append(part_ious)
        test_writer.add_summary(summary, step)

    # Add summaries
    shape_ious = np.array(shape_ious)
    test_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Seg_IOU/background_new', simple_value=np.mean(shape_ious[:,0]) )]), EPOCH_CNT)
    test_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Seg_IOU/instance_new', simple_value=np.mean(shape_ious[:,1]) )]), EPOCH_CNT)
    test_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Seg_IOU/Mean', simple_value=np.mean(shape_ious) )]), EPOCH_CNT)

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % \
        (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    log_string('eval mIoU: %f' % (np.mean(shape_ious)))
    log_string('eval box IoU (ground/3D)     : %f / %f' % 
        (iou2ds_sum / float(num_batches*BATCH_SIZE), iou3ds_sum / float(num_batches*BATCH_SIZE)))
       
    EPOCH_CNT += 1
    return loss_sum/float(num_batches)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
