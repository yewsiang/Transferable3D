
""" Configuration file for scripts.

Author: Tang Yew Siang
Date: July 2019
"""

import os
import sys
import argparse
import numpy as np

# =============================== Algorithm Details ===============================
# A: Combines the PC with the 3D box by using planes of the 3D box
# B: Combines the PC with the 3D box by MLPs into a higher dims feature space
BOX_PC_MASK_REPRESENTATION           = '' # A/B
USE_NORMALIZED_BOX2D_AS_FEATS        = False
NORMALIZE_PC_BEFORE_SEG              = False
NORMALIZATION_METHOD                 = '' # SD/Spread

# ===================== Box PC Fit ====================
BOXPC_SAMPLING_METHOD                = 'SAMPLE' # BATCH / SAMPLE
BOXPC_SAMPLE_EQUAL_CLASS_WITH_PROB   = 1.
BOXPC_PROPORTION_OF_BOXPC_FIT        = 0.5
BOXPC_NOFIT_BOUNDS                   = [0.01, 0.25]
BOXPC_FIT_BOUNDS                     = [0.7, 1.0]
BOXPC_CENTER_PERTURBATION            = 0.8
BOXPC_SIZE_PERTURBATION              = 0.2
BOXPC_ANGLE_PERTURBATION             = np.pi

# Use Box fit cls prediction to weigh delta prediction / loss
BOXPC_DELTA_LOSS_TYPE                = 'huber' # mse / huber. Specific to B
BOXPC_WEIGH_DELTA_PRED_BY_CLS_CONF   = False # Specific to B
BOXPC_WEIGH_DELTA_LOSS_BY_CLS_CONF   = False # Specific to B
BOXPC_STOP_GRAD_OF_CLS_VIA_DELTA     = True  # Specific to B
# Use GT cls to weigh delta loss
BOXPC_WEIGH_DELTA_LOSS_BY_CLS_GT     = False # Specific to B

# Weights
BOXPC_WEIGHT_CLS                     = 1.
BOXPC_WEIGHT_DELTA                   = 1.   # Specific to B
BOXPC_WEIGHT_DELTA_CENTER_PERCENT    = 0.34 # Specific to B
BOXPC_WEIGHT_DELTA_SIZE_PERCENT      = 0.33 # Specific to B
BOXPC_WEIGHT_DELTA_ANGLE_PERCENT     = 0.33 # Specific to B
BOXPC_WEIGHT_CLUSTER                 = 1.   # Specific to B & C

# ================== Semi Supervised ==================
SEMI_MODEL                           = '' # A/B/C/D
# Sampling method, options: 
# BATCH           : Simply iterate through the dataset
# ALTERNATE_BATCH : Alternate between Weak and Strong samples + Sample
# MIXED_BATCH     : Mix Weak and Strong samples + Sample
SEMI_SAMPLING_METHOD                 = 'ALTERNATE_BATCH'
# Probability of sampling equal class, else sample from data distribution
# If set to 1, it will randomly sample an equal num of samples / class
# If set to 0, it will randomly sample from the dataset
SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB    = 0.    # ********* IMPORTANT **********
SEMI_USE_LABELS2D_OF_CLASSES3D       = False # ********* IMPORTANT **********

# Adversarial loss related parameters
SEMI_ADV_INITIAL_ITERS_BEFORE_TRAIN  = 0
SEMI_ADV_INITIAL_TRAINING_EPOCHS     = 0
SEMI_ADV_ITERS_FOR_D                 = 0 # 1

SEMI_ADV_SAMPLE_EQUAL_CLASS_W_PROB   = 0.    # TODO
SEMI_ADV_DROPOUTS_FOR_G              = 0.5 
SEMI_ADV_FLIP_LABELS_FOR_D_PROB      = 0. 
SEMI_ADV_SOFT_NOISY_LABELS_FOR_D     = False 
SEMI_ADV_FEATURE_MATCHING            = False # TODO
SEMI_ADV_NORMALIZE_PC_TO_NEG1_TO_1   = False # NA
SEMI_ADV_TANH_FOR_LAST_LAYER_OF_G    = True 
SEMI_ADV_DIFF_MINIBATCH_REAL_VS_FAKE = False # TODO
SEMI_ADV_LEAKY_RELU                  = True 
SEMI_ADV_AVERAGE_POOLING             = False # NA

SEMI_TRAIN_BOXPC_MODEL               = False
SEMI_TRAIN_SEG_TRAIN_CLASS_AG_SEG    = False
SEMI_TRAIN_BOX_TRAIN_CLASS_AG_TNET   = False
SEMI_TRAIN_BOX_TRAIN_CLASS_AG_BOX    = False
# Box-PC related
# Min. the Box PC Fit loss AFTER refining instead of before
SEMI_BOXPC_MIN_FIT_LOSS_AFT_REFINE   = False
SEMI_BOXPC_FIT_ONLY_ON_2D_CLS        = False
SEMI_WEIGH_BOXPC_DELTA_DURING_TEST   = False
SEMI_REFINE_USING_BOXPC_DELTA_NUM    = 1
# Intracls Dims Var
SEMI_INTRACLSDIMS_ONLY_ON_2D_CLS     = True

# ================================= Loss Details =================================
# ========= Semi Supervised =========
SEMI_MULTIPLIER_FOR_WEAK_LOSS        = 1 #0.1
SEMI_WEIGHT_REG_DELTA_LOSS           = 0.
SEMI_WEIGHT_G_LOSS                   = 0.4
SEMI_WEIGHT_G_FEATURE_MATCH_LOSS     = 0.
SEMI_WEIGHT_BOXPC_FIT_LOSS           = 1.
SEMI_WEIGHT_BOXPC_INTRACLS_FIT_LOSS  = 0 #0.001
SEMI_WEIGHT_BOXPC_KMEANS_LOSS        = 0

# ======== Weakly Supervised ========
# Loss Weights
# 1. Segmentation
WEAK_WEIGHT_CROSS_ENTROPY            = 5.
WEAK_CLS_WEIGHTS_CROSS_ENTROPY       = [2., 0.333, 1., 0.333]
WEAK_WEIGHT_VARIANCE                 = 1.
WEAK_WEIGHT_INACTIVE                 = 4.
WEAK_WEIGHT_BINARY                   = 0.002
# 2. Box Estimation
WEAK_WEIGHT_ANCHOR_CLS               = 1.
# Variance of y center predictions across the batch
WEAK_WEIGHT_CENTER_YVAR              = 1.
WEAK_WEIGHT_INACTIVE_VOLUME          = 0 #1.
WEAK_WEIGHT_REPROJECTION             = 0.01 #0.25
WEAK_WEIGHT_SURFACE                  = 1.
WEAK_WEIGHT_INTRACLASSVAR            = 0 #0.025

# Whether to train Seg / Box with the weak losses.
# Box parameters to train each of the following loss functions (Center, Dims, Orient)
# If False, tf.stop_gradient will be applied to prevent gradients from propagating. None means N.A.
WEAK_TRAIN_SEG_W_SURFACE             = False
WEAK_TRAIN_BOX_W_REPROJECTION        = [True, True, True]
WEAK_TRAIN_BOX_W_SURFACE             = [True, False, True]

# Margin for Variance loss before there will be no more penalization (to prevent excessively small masks)
# A margin of 1 means that once the mask lies within a sphere of radius 1m, it will no longer be penalized.
# A margin of 0 means there is no margin, it will always want to seek a smaller mask.
WEAK_VARIANCE_LOSS_MARGIN            = 1.5

# Whether to use the ground truth anchor label
WEAK_USE_GT_ANCHOR_CLS_LABEL         = False

# Margin for Inactive vol loss before there will be no more penalization (to prevent excessively small vols)
WEAK_INACTIVE_VOL_LOSS_MARGINS       = [10.,0.,0.]
WEAK_INACTIVE_VOL_ONLY_ON_2D_CLS     = True

# Apply softmax on projection instead of maximum
# Depending on softmax_scale, the points that are closest to the max/min will be given different weightings
# EG. Suppose there are 8 points [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0], after softmax, the weightings are:
# Weighting given for scale = 1:  [0.071, 0.079, 0.087, 0.106, 0.13 , 0.158, 0.175, 0.194]
# Weighting given for scale = 5:  [0.003, 0.005, 0.008, 0.023, 0.062, 0.168, 0.276, 0.455]
# Weighting given for scale = 10: [0.0  , 0.0  , 0.0  , 0.002, 0.012, 0.089, 0.241, 0.656]
# Weighting given for scale = 20: [0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.016, 0.117, 0.867]
# Weighting given for scale = 40: [0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.018, 0.982]
WEAK_REPROJECTION_USE_SOFTMAX_PROJ   = False
WEAK_REPROJECTION_SOFTMAX_SCALE      = 10.
# Only apply on 2D classes or 2D+3D classes
WEAK_REPROJECTION_ONLY_ON_2D_CLS     = False
# Whether the reprojection loss should be applied on all points
WEAK_REPROJECTION_CLIP_LOWERB_LOSS   = True
# Whether to clip the projection of the 3D bbox or not (will prevent training if it exceeds the box)
WEAK_REPROJECTION_CLIP_PRED_BOX      = False
# Reprojection Loss (huber/mse)
WEAK_REPROJECTION_LOSS_TYPE          = 'huber'
# Given the GT 2D bbox as the lower bound, we will dilate it by this factor to give the upper bound.
# The reprojection loss does not penalize for Pred 2D bboxes between these bounds.
WEAK_REPROJECTION_DILATE_FACTOR      = 1.5

# No penalization if points lie within margin
WEAK_SURFACE_MARGIN                  = 0
# If points are within the box, surface loss will be weighted
WEAK_SURFACE_LOSS_WT_FOR_INNER_PTS   = 0.8
# Value to scale dims before calculating surface loss
WEAK_SURFACE_LOSS_SCALE_DIMS         = 0.9

# Intraclass Dims Variance Loss (huber/mse)
WEAK_DIMS_LOSS_TYPE                  = 'huber'
# Whether to use margin for loss or not
WEAK_DIMS_USE_MARGIN_LOSS            = True
# Margin allowed before penalizing
WEAK_DIMS_SD_MARGIN                  = 0.2
# Decay to be applies to the exponential moving average of dims per class
WEAK_DIMS_EMA_DECAY                  = 0.99
 
# ======== Strongly Supervised ========
# Loss Weights
# 1. Segmentation
STRONG_WEIGHT_CROSS_ENTROPY          = 1.

# 2. Box Estimation
STRONG_BOX_MULTIPLER                 = 0.1
STRONG_WEIGHT_CENTER                 = 1.
STRONG_WEIGHT_ORIENT_CLS             = 1.
STRONG_WEIGHT_ORIENT_REG             = 20.
STRONG_WEIGHT_DIMS_CLS               = 1.
STRONG_WEIGHT_DIMS_REG               = 20.
STRONG_WEIGHT_TNET_CENTER            = 1.
STRONG_WEIGHT_CORNER                 = 1.

# ==================================== SUNRGBD ====================================
# Classes when doing strongly supervised learning
SUNRGBD_STRONG_TRAIN_CLS = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']

# Classes when doing semi supervised learning
SUNRGBD_SEMI_TRAIN_CLS   = ['bed','chair','toilet','desk','bathtub']
SUNRGBD_SEMI_TEST_CLS    = [cls_type for cls_type in SUNRGBD_STRONG_TRAIN_CLS if cls_type not in SUNRGBD_SEMI_TRAIN_CLS]

# Classes when doing weakly supervised learning
SUNRGBD_WEAK_TRAIN_CLS   = SUNRGBD_STRONG_TRAIN_CLS
SUNRGBD_WEAK_TEST_CLS    = SUNRGBD_STRONG_TRAIN_CLS

# ===================================== KITTI =====================================
KITTI_ALL_CLS            = ['Car', 'Pedestrian', 'Cyclist']

# Classes when doing semi supervised learning
KITTI_SEMI_TRAIN_CLS     = [] #['Car', 'Pedestrian']
KITTI_SEMI_TEST_CLS      = [cls_type for cls_type in KITTI_ALL_CLS if cls_type not in KITTI_SEMI_TRAIN_CLS]


class SpecialArgumentParser(argparse.ArgumentParser):
    """
    Works like standard Argparse but provides set_attributes function to support adding in of dictionary / array
    attributes in the object returned by parse_special_args.
    """
    def __init__(self):
        argparse.ArgumentParser.__init__(self)
        self.attr_names_and_vals = None

    def parse_special_args(self):
        flags = self.parse_args()
        for attr_name, attr_val in self.attr_names_and_vals:
            setattr(flags, attr_name, attr_val)
        config_str = self.get_config_str(flags)
        setattr(flags, 'config_str', config_str)
        return flags

    def set_attributes(self, attr_names_and_vals):
        self.attr_names_and_vals = attr_names_and_vals

    def get_config_str(self, c):
        if not hasattr(c, 'mode'): c.mode = None
        config_str = '\n * REMEMBER TO CHECK THE CONFIGURATIONS * \n\n' + \
        '  [MODE: %s]\n' % (c.mode) + \
        '  [CLASSES]\n' + \
        '    SUNRGBD_STRONG_TRAIN_CLS                : %s\n' % (c.SUNRGBD_STRONG_TRAIN_CLS) + \
        '    SUNRGBD_SEMI_TRAIN_CLS                  : %s\n' % (c.SUNRGBD_SEMI_TRAIN_CLS) + \
        '    SUNRGBD_SEMI_TEST_CLS                   : %s\n' % (c.SUNRGBD_SEMI_TEST_CLS) + \
        '    SUNRGBD_WEAK_TRAIN_CLS                  : %s\n' % (c.SUNRGBD_WEAK_TRAIN_CLS) + \
        '    SUNRGBD_WEAK_TEST_CLS                   : %s\n' % (c.SUNRGBD_WEAK_TEST_CLS)
        return config_str

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

cfg = SpecialArgumentParser()
cfg.set_attributes([])
# Parameters that can be changed using the command line
cfg.add_argument('--BOX_PC_MASK_REPRESENTATION', type=str, default=BOX_PC_MASK_REPRESENTATION)
cfg.add_argument('--USE_NORMALIZED_BOX2D_AS_FEATS', type=str2bool, default=USE_NORMALIZED_BOX2D_AS_FEATS)
cfg.add_argument('--NORMALIZE_PC_BEFORE_SEG', type=str2bool, default=NORMALIZE_PC_BEFORE_SEG)
cfg.add_argument('--NORMALIZATION_METHOD', type=str, default=NORMALIZATION_METHOD)
cfg.add_argument('--BOXPC_SAMPLING_METHOD', type=str, default=BOXPC_SAMPLING_METHOD)
cfg.add_argument('--BOXPC_SAMPLE_EQUAL_CLASS_WITH_PROB', type=float, default=BOXPC_SAMPLE_EQUAL_CLASS_WITH_PROB)
cfg.add_argument('--BOXPC_PROPORTION_OF_BOXPC_FIT', type=float, default=BOXPC_PROPORTION_OF_BOXPC_FIT)
cfg.add_argument('--BOXPC_NOFIT_BOUNDS', nargs='+', type=float, default=BOXPC_NOFIT_BOUNDS)
cfg.add_argument('--BOXPC_FIT_BOUNDS', nargs='+', type=float, default=BOXPC_FIT_BOUNDS)
cfg.add_argument('--BOXPC_CENTER_PERTURBATION', type=float, default=BOXPC_CENTER_PERTURBATION)
cfg.add_argument('--BOXPC_SIZE_PERTURBATION', type=float, default=BOXPC_SIZE_PERTURBATION)
cfg.add_argument('--BOXPC_ANGLE_PERTURBATION', type=float, default=BOXPC_ANGLE_PERTURBATION)
cfg.add_argument('--BOXPC_DELTA_LOSS_TYPE', type=str, default=BOXPC_DELTA_LOSS_TYPE)
cfg.add_argument('--BOXPC_WEIGH_DELTA_PRED_BY_CLS_CONF', type=str2bool, default=BOXPC_WEIGH_DELTA_PRED_BY_CLS_CONF)
cfg.add_argument('--BOXPC_WEIGH_DELTA_LOSS_BY_CLS_CONF', type=str2bool, default=BOXPC_WEIGH_DELTA_LOSS_BY_CLS_CONF)
cfg.add_argument('--BOXPC_STOP_GRAD_OF_CLS_VIA_DELTA', type=str2bool, default=BOXPC_STOP_GRAD_OF_CLS_VIA_DELTA)
cfg.add_argument('--BOXPC_WEIGH_DELTA_LOSS_BY_CLS_GT', type=str2bool, default=BOXPC_WEIGH_DELTA_LOSS_BY_CLS_GT)
cfg.add_argument('--BOXPC_WEIGHT_CLS', type=float, default=BOXPC_WEIGHT_CLS)
cfg.add_argument('--BOXPC_WEIGHT_DELTA', type=float, default=BOXPC_WEIGHT_DELTA)
cfg.add_argument('--BOXPC_WEIGHT_DELTA_CENTER_PERCENT', type=float, default=BOXPC_WEIGHT_DELTA_CENTER_PERCENT)
cfg.add_argument('--BOXPC_WEIGHT_DELTA_SIZE_PERCENT', type=float, default=BOXPC_WEIGHT_DELTA_SIZE_PERCENT)
cfg.add_argument('--BOXPC_WEIGHT_DELTA_ANGLE_PERCENT', type=float, default=BOXPC_WEIGHT_DELTA_ANGLE_PERCENT)
cfg.add_argument('--BOXPC_WEIGHT_CLUSTER', type=float, default=BOXPC_WEIGHT_CLUSTER)
cfg.add_argument('--SEMI_MODEL', type=str, default=SEMI_MODEL)
cfg.add_argument('--SEMI_SAMPLING_METHOD', type=str, default=SEMI_SAMPLING_METHOD)
cfg.add_argument('--SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB', type=float, default=SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB)
cfg.add_argument('--SEMI_USE_LABELS2D_OF_CLASSES3D', type=str2bool, default=SEMI_USE_LABELS2D_OF_CLASSES3D)
cfg.add_argument('--SEMI_ADV_INITIAL_ITERS_BEFORE_TRAIN', type=int, default=SEMI_ADV_INITIAL_ITERS_BEFORE_TRAIN)
cfg.add_argument('--SEMI_ADV_INITIAL_TRAINING_EPOCHS', type=int, default=SEMI_ADV_INITIAL_TRAINING_EPOCHS)
cfg.add_argument('--SEMI_ADV_ITERS_FOR_D', type=int, default=SEMI_ADV_ITERS_FOR_D)
cfg.add_argument('--SEMI_ADV_SAMPLE_EQUAL_CLASS_W_PROB', type=float, default=SEMI_ADV_SAMPLE_EQUAL_CLASS_W_PROB)
cfg.add_argument('--SEMI_ADV_DROPOUTS_FOR_G', type=float, default=SEMI_ADV_DROPOUTS_FOR_G)
cfg.add_argument('--SEMI_ADV_FLIP_LABELS_FOR_D_PROB', type=float, default=SEMI_ADV_FLIP_LABELS_FOR_D_PROB)
cfg.add_argument('--SEMI_ADV_SOFT_NOISY_LABELS_FOR_D', type=str2bool, default=SEMI_ADV_SOFT_NOISY_LABELS_FOR_D)
cfg.add_argument('--SEMI_ADV_FEATURE_MATCHING', type=str2bool, default=SEMI_ADV_FEATURE_MATCHING)
cfg.add_argument('--SEMI_ADV_NORMALIZE_PC_TO_NEG1_TO_1', type=str2bool, default=SEMI_ADV_NORMALIZE_PC_TO_NEG1_TO_1)
cfg.add_argument('--SEMI_ADV_TANH_FOR_LAST_LAYER_OF_G', type=str2bool, default=SEMI_ADV_TANH_FOR_LAST_LAYER_OF_G)
cfg.add_argument('--SEMI_ADV_DIFF_MINIBATCH_REAL_VS_FAKE', type=str2bool, default=SEMI_ADV_DIFF_MINIBATCH_REAL_VS_FAKE)
cfg.add_argument('--SEMI_ADV_LEAKY_RELU', type=str2bool, default=SEMI_ADV_LEAKY_RELU)
cfg.add_argument('--SEMI_ADV_AVERAGE_POOLING', type=str2bool, default=SEMI_ADV_AVERAGE_POOLING)
cfg.add_argument('--SEMI_TRAIN_BOXPC_MODEL', type=str2bool, default=SEMI_TRAIN_BOXPC_MODEL)
cfg.add_argument('--SEMI_TRAIN_SEG_TRAIN_CLASS_AG_SEG', type=str2bool, default=SEMI_TRAIN_SEG_TRAIN_CLASS_AG_SEG)
cfg.add_argument('--SEMI_TRAIN_BOX_TRAIN_CLASS_AG_TNET', type=str2bool, default=SEMI_TRAIN_BOX_TRAIN_CLASS_AG_TNET)
cfg.add_argument('--SEMI_TRAIN_BOX_TRAIN_CLASS_AG_BOX', type=str2bool, default=SEMI_TRAIN_BOX_TRAIN_CLASS_AG_BOX)
cfg.add_argument('--SEMI_BOXPC_MIN_FIT_LOSS_AFT_REFINE', type=str2bool, default=SEMI_BOXPC_MIN_FIT_LOSS_AFT_REFINE)
cfg.add_argument('--SEMI_BOXPC_FIT_ONLY_ON_2D_CLS', type=str2bool, default=SEMI_BOXPC_FIT_ONLY_ON_2D_CLS)
cfg.add_argument('--SEMI_WEIGH_BOXPC_DELTA_DURING_TEST', type=str2bool, default=SEMI_WEIGH_BOXPC_DELTA_DURING_TEST)
cfg.add_argument('--SEMI_REFINE_USING_BOXPC_DELTA_NUM', type=int, default=SEMI_REFINE_USING_BOXPC_DELTA_NUM)
cfg.add_argument('--SEMI_INTRACLSDIMS_ONLY_ON_2D_CLS', type=str2bool, default=SEMI_INTRACLSDIMS_ONLY_ON_2D_CLS)
cfg.add_argument('--SEMI_MULTIPLIER_FOR_WEAK_LOSS', type=float, default=SEMI_MULTIPLIER_FOR_WEAK_LOSS)
cfg.add_argument('--SEMI_WEIGHT_REG_DELTA_LOSS', type=float, default=SEMI_WEIGHT_REG_DELTA_LOSS)
cfg.add_argument('--SEMI_WEIGHT_G_LOSS', type=float, default=SEMI_WEIGHT_G_LOSS)
cfg.add_argument('--SEMI_WEIGHT_G_FEATURE_MATCH_LOSS', type=float, default=SEMI_WEIGHT_G_FEATURE_MATCH_LOSS)
cfg.add_argument('--SEMI_WEIGHT_BOXPC_FIT_LOSS', type=float, default=SEMI_WEIGHT_BOXPC_FIT_LOSS)
cfg.add_argument('--SEMI_WEIGHT_BOXPC_INTRACLS_FIT_LOSS', type=float, default=SEMI_WEIGHT_BOXPC_INTRACLS_FIT_LOSS)
cfg.add_argument('--SEMI_WEIGHT_BOXPC_KMEANS_LOSS', type=float, default=SEMI_WEIGHT_BOXPC_KMEANS_LOSS)
cfg.add_argument('--WEAK_WEIGHT_CROSS_ENTROPY', type=float, default=WEAK_WEIGHT_CROSS_ENTROPY)
cfg.add_argument('--WEAK_CLS_WEIGHTS_CROSS_ENTROPY', nargs='+', type=float, default=WEAK_CLS_WEIGHTS_CROSS_ENTROPY)
cfg.add_argument('--WEAK_WEIGHT_VARIANCE', type=float, default=WEAK_WEIGHT_VARIANCE)
cfg.add_argument('--WEAK_WEIGHT_INACTIVE', type=float, default=WEAK_WEIGHT_INACTIVE)
cfg.add_argument('--WEAK_WEIGHT_BINARY', type=float, default=WEAK_WEIGHT_BINARY)
cfg.add_argument('--WEAK_WEIGHT_ANCHOR_CLS', type=float, default=WEAK_WEIGHT_ANCHOR_CLS)
cfg.add_argument('--WEAK_WEIGHT_CENTER_YVAR', type=float, default=WEAK_WEIGHT_CENTER_YVAR)
cfg.add_argument('--WEAK_WEIGHT_INACTIVE_VOLUME', type=float, default=WEAK_WEIGHT_INACTIVE_VOLUME)
cfg.add_argument('--WEAK_WEIGHT_REPROJECTION', type=float, default=WEAK_WEIGHT_REPROJECTION)
cfg.add_argument('--WEAK_WEIGHT_SURFACE', type=float, default=WEAK_WEIGHT_SURFACE)
cfg.add_argument('--WEAK_WEIGHT_INTRACLASSVAR', type=float, default=WEAK_WEIGHT_INTRACLASSVAR)
cfg.add_argument('--WEAK_TRAIN_SEG_W_SURFACE', type=str2bool, default=WEAK_TRAIN_SEG_W_SURFACE)
cfg.add_argument('--WEAK_TRAIN_BOX_W_REPROJECTION', nargs='+', type=str2bool, default=WEAK_TRAIN_BOX_W_REPROJECTION)
cfg.add_argument('--WEAK_TRAIN_BOX_W_SURFACE', nargs='+', type=str2bool, default=WEAK_TRAIN_BOX_W_SURFACE)
cfg.add_argument('--WEAK_VARIANCE_LOSS_MARGIN', type=float, default=WEAK_VARIANCE_LOSS_MARGIN)
cfg.add_argument('--WEAK_USE_GT_ANCHOR_CLS_LABEL', type=str2bool, default=WEAK_USE_GT_ANCHOR_CLS_LABEL)
cfg.add_argument('--WEAK_INACTIVE_VOL_LOSS_MARGINS', nargs='+', type=float, default=WEAK_INACTIVE_VOL_LOSS_MARGINS)
cfg.add_argument('--WEAK_INACTIVE_VOL_ONLY_ON_2D_CLS', type=str2bool, default=WEAK_INACTIVE_VOL_ONLY_ON_2D_CLS)
cfg.add_argument('--WEAK_REPROJECTION_USE_SOFTMAX_PROJ', type=str2bool, default=WEAK_REPROJECTION_USE_SOFTMAX_PROJ)
cfg.add_argument('--WEAK_REPROJECTION_SOFTMAX_SCALE', type=str2bool, default=WEAK_REPROJECTION_SOFTMAX_SCALE)
cfg.add_argument('--WEAK_REPROJECTION_ONLY_ON_2D_CLS', type=str2bool, default=WEAK_REPROJECTION_ONLY_ON_2D_CLS)
cfg.add_argument('--WEAK_REPROJECTION_CLIP_LOWERB_LOSS', type=str2bool, default=WEAK_REPROJECTION_CLIP_LOWERB_LOSS)
cfg.add_argument('--WEAK_REPROJECTION_CLIP_PRED_BOX', type=str2bool, default=WEAK_REPROJECTION_CLIP_PRED_BOX)
cfg.add_argument('--WEAK_REPROJECTION_LOSS_TYPE', type=str, default=WEAK_REPROJECTION_LOSS_TYPE)
cfg.add_argument('--WEAK_REPROJECTION_DILATE_FACTOR', type=float, default=WEAK_REPROJECTION_DILATE_FACTOR)
cfg.add_argument('--WEAK_SURFACE_MARGIN', type=float, default=WEAK_SURFACE_MARGIN)
cfg.add_argument('--WEAK_SURFACE_LOSS_WT_FOR_INNER_PTS', type=float, default=WEAK_SURFACE_LOSS_WT_FOR_INNER_PTS)
cfg.add_argument('--WEAK_SURFACE_LOSS_SCALE_DIMS', type=float, default=WEAK_SURFACE_LOSS_SCALE_DIMS)
cfg.add_argument('--WEAK_DIMS_LOSS_TYPE', type=str, default=WEAK_DIMS_LOSS_TYPE)
cfg.add_argument('--WEAK_DIMS_USE_MARGIN_LOSS', type=str2bool, default=WEAK_DIMS_USE_MARGIN_LOSS)
cfg.add_argument('--WEAK_DIMS_SD_MARGIN', type=float, default=WEAK_DIMS_SD_MARGIN)
cfg.add_argument('--WEAK_DIMS_EMA_DECAY', type=float, default=WEAK_DIMS_EMA_DECAY)
cfg.add_argument('--STRONG_WEIGHT_CROSS_ENTROPY', type=float, default=STRONG_WEIGHT_CROSS_ENTROPY)
cfg.add_argument('--STRONG_BOX_MULTIPLER', type=float, default=STRONG_BOX_MULTIPLER)
cfg.add_argument('--STRONG_WEIGHT_CENTER', type=float, default=STRONG_WEIGHT_CENTER)
cfg.add_argument('--STRONG_WEIGHT_ORIENT_CLS', type=float, default=STRONG_WEIGHT_ORIENT_CLS)
cfg.add_argument('--STRONG_WEIGHT_ORIENT_REG', type=float, default=STRONG_WEIGHT_ORIENT_REG)
cfg.add_argument('--STRONG_WEIGHT_DIMS_CLS', type=float, default=STRONG_WEIGHT_DIMS_CLS)
cfg.add_argument('--STRONG_WEIGHT_DIMS_REG', type=float, default=STRONG_WEIGHT_DIMS_REG)
cfg.add_argument('--STRONG_WEIGHT_TNET_CENTER', type=float, default=STRONG_WEIGHT_TNET_CENTER)
cfg.add_argument('--STRONG_WEIGHT_CORNER', type=float, default=STRONG_WEIGHT_CORNER)
cfg.add_argument('--SUNRGBD_STRONG_TRAIN_CLS', nargs='+', type=str, default=SUNRGBD_STRONG_TRAIN_CLS)
cfg.add_argument('--SUNRGBD_SEMI_TRAIN_CLS', nargs='+', type=str, default=SUNRGBD_SEMI_TRAIN_CLS)
cfg.add_argument('--SUNRGBD_SEMI_TEST_CLS', nargs='+', type=str, default=SUNRGBD_SEMI_TEST_CLS)
cfg.add_argument('--SUNRGBD_WEAK_TRAIN_CLS', nargs='+', type=str, default=SUNRGBD_WEAK_TRAIN_CLS)
cfg.add_argument('--SUNRGBD_WEAK_TEST_CLS', nargs='+', type=str, default=SUNRGBD_WEAK_TEST_CLS)
cfg.add_argument('--KITTI_ALL_CLS', nargs='+', type=str, default=KITTI_ALL_CLS)
cfg.add_argument('--KITTI_SEMI_TRAIN_CLS', nargs='+', type=str, default=KITTI_SEMI_TRAIN_CLS)
cfg.add_argument('--KITTI_SEMI_TEST_CLS', nargs='+', type=str, default=KITTI_SEMI_TEST_CLS)
