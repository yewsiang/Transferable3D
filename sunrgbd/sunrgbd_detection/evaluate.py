
""" Evaluation utilities.

Author: Tang Yew Siang
Date: July 2019
"""

import os
import sys
import argparse
import cPickle as pickle
from os.path import join as pjoin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../sunrgbd_data'))
sys.path.append(os.path.join(BASE_DIR, '../../train'))

import numpy as np
import roi_seg_box3d_dataset
from eval_det import eval_det
from box_util import box3d_iou
from sunrgbd_data import sunrgbd_object
from utils import rotz, compute_box_3d, load_zipped_pickle
from roi_seg_box3d_dataset import rotate_pc_along_y, NUM_HEADING_BIN


def evaluate_predictions(predictions, dataset, classes, test_dataset, test_classes):
    ps_list, _, segp_list, center_list, heading_cls_list, heading_res_list, size_cls_list, size_res_list, \
    rot_angle_list, score_list, cls_list, file_num_list, _, _ = predictions
    print('Number of PRED boxes: %d, Number of GT boxes: %d' % (len(segp_list), len(test_dataset)))

    # For detection evaluation
    pred_all = {}
    gt_all = {}
    ovthresh = 0.25

    # A) Get GT boxes
    # print('Constructing GT boxes...')
    for i in range(len(test_dataset)):
        img_id = test_dataset.idx_l[i]
        if img_id in gt_all: continue # All ready counted..
        gt_all[img_id] = []

        objects = dataset.get_label_objects(img_id)
        calib = dataset.get_calibration(img_id)
        for obj in objects:
            if obj.classname not in test_classes: continue
            box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib)
            box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
            box3d_pts_3d_flipped = np.copy(box3d_pts_3d)
            box3d_pts_3d_flipped[0:4,:] = box3d_pts_3d[4:,:]
            box3d_pts_3d_flipped[4:,:] = box3d_pts_3d[0:4,:]
            gt_all[img_id].append((obj.classname, box3d_pts_3d_flipped))

    # B) Get PRED boxes
    # print('Constructing PRED boxes...')
    for i in range(len(ps_list)):
        img_id = file_num_list[i] 
        classname = classes[cls_list[i]]
        if classname not in test_classes: raise Exception('Not supposed to have class: %s' % classname)
        center = center_list[i].squeeze()
        rot_angle = rot_angle_list[i]

        # Get heading angle and size
        heading_angle = roi_seg_box3d_dataset.class2angle(heading_cls_list[i], heading_res_list[i], NUM_HEADING_BIN)
        box_size = roi_seg_box3d_dataset.class2size(size_cls_list[i], size_res_list[i]) 
        corners_3d_pred = roi_seg_box3d_dataset.get_3d_box(box_size, heading_angle, center)
        corners_3d_pred = rotate_pc_along_y(corners_3d_pred, -rot_angle)

        if img_id not in pred_all:
            pred_all[img_id] = []
        pred_all[img_id].append((classname, corners_3d_pred, score_list[i]))

    # Compute AP
    rec, prec, ap = eval_det(pred_all, gt_all, ovthresh)
    mean_ap = np.mean([ap[classname] for classname in ap])

    return rec, prec, ap, mean_ap

def plot_ap(ap):
    # C) Evaluate
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rc('axes', linewidth=2)

    fig = plt.figure()
    N_figs = len(ap.keys())
    for i, classname in enumerate(ap.keys()):
        fig.subplots_adjust(wspace=0.5, hspace=1.0)
        ax1 = fig.add_subplot(N_figs // 2, 2, i+1)
        ax1.plot(rec[classname], prec[classname], lw=3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=24)
        plt.ylabel('Precision', fontsize=24)
        plt.title(classname, fontsize=24)
    plt.show()

def get_ap_info(ap, mean_ap):
    string = 'Average Precision:\n'
    sorted_keys = ap.keys()
    sorted_keys.sort()
    for classname in sorted_keys:
        string += ('%11s: [%.1f]\n' % (classname, 100. * ap[classname]))
    string += ('    Mean AP:  %.1f' % (100. * mean_ap))
    return string

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', default=None, help='GT path for .pickle file, the one used for val in train.py [default: None]')
    parser.add_argument('--pred_path', default=None, help='Prediction path for .pickle file from test.py [default: None]')
    FLAGS = parser.parse_args()

    CLASSES      = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
    TEST_CLASSES = ['table','sofa','dresser','night_stand','bookshelf'] #['bed','chair','toilet','desk','bathtub']
    SUNRGBD_DATASET_DIR = '/home/yewsiang/Transferable3D/dataset/mysunrgbd'
    IMG_DIR = pjoin(SUNRGBD_DATASET_DIR, 'training', 'image')

    # Load GT
    TEST_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(TEST_CLASSES, npoints=2048, 
                                                          split='val', rotate_to_center=True, 
                                                          overwritten_data_path=FLAGS.gt_path, 
                                                          from_rgb_detection=False)
    DATASET = sunrgbd_object(SUNRGBD_DATASET_DIR, 'training')
    
    # Load predictions
    predictions = load_zipped_pickle(FLAGS.pred_path)
    rec, prec, ap, mean_ap = evaluate_predictions(predictions, DATASET, CLASSES, TEST_DATASET, TEST_CLASSES)
    print(get_ap_info(ap, mean_ap))
    plot_ap(ap)