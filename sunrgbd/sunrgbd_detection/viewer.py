
""" Visualization of point clouds, predictions, datasets.

Author: Tang Yew Siang
Date: July 2019
"""

import os
import sys
import random
import numpy as np
from PIL import Image
from os.path import join as pjoin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, '../sunrgbd_data'))
sys.path.append(os.path.join(BASE_DIR, '../../train'))
sys.path.append(os.path.join(BASE_DIR, '../../models'))

import vis_utils as vis
from config import cfg
from box_util import box3d_iou
from box_pc_fit_dataset import BoxPCFitDataset
from sunrgbd_data import sunrgbd_object, load_data, process_object
from utils import rotz, compute_box_3d, load_zipped_pickle, extract_pc_in_box3d
from roi_seg_box3d_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, ROISegBoxDataset, rotate_pc_along_y, get_3d_box, class2size, class2angle, class2type

cfg.add_argument('--seed', type=int, default=np.random.randint(1e6))
cfg.add_argument('--vis', required=True, help='Type of visualization to show', 
                 choices=['pc', 'fpc', 'seg_box', 'box_pc', 'pred2d', 'pred3d'])
cfg.add_argument('--filename', help='File name to show', type=str)
cfg.add_argument('--filenum', help='File number to show', type=int)
cfg.add_argument('--filenums', help='Only show these file numbers', nargs='+', type=int)
cfg.add_argument('--pred_files', help='Prediction file', nargs='+', type=str)
cfg.add_argument('--gt_file', help='Ground truth file for pred3d', type=str)
cfg.add_argument('--num', help='Number to show', type=int, default=10)
cfg.add_argument('--rgb_detection', action='store_true')
FLAGS = cfg.parse_special_args()
np.random.seed(FLAGS.seed)
print('Seed value: %d' % FLAGS.seed)

CLASSES    = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
# WHITE_LIST = CLASSES # ['bed','chair','toilet','desk','bathtub']
WHITE_LIST = ['table','sofa','dresser','night_stand','bookshelf'] 
SUNRGBD_DATASET_DIR = '/home/yewsiang/Transferable3D/dataset/mysunrgbd'
IMG_DIR = pjoin(SUNRGBD_DATASET_DIR, 'training', 'image')
dataset = sunrgbd_object(SUNRGBD_DATASET_DIR, 'training')

# Visualization parameters
SEP        = 6.0
COLS       = 8 # Number of columns for visualization
IMG_SCALE  = 0.005 # Image scale
TEXT_SCALE = (0.035,0.035,0.035)
TEXT_ROT   = (0,180,180)


def vis_point_cloud(file_num):
    # Get dataset information
    calib, objects, pc_upright_camera, img, pc_image_coord = load_data(dataset, file_num)

    gt_box3d_list, gt_box2d_list, gt_cls_list = [], [], []
    for obj in objects:
        if obj.classname not in WHITE_LIST: continue
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib)
        box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
        gt_box3d_list.append(box3d_pts_3d)
        gt_box2d_list.append(obj.box2d)
        gt_cls_list.append(obj.classname)

    # Visualize point cloud
    vtk_actors = []
    vtk_gt_boxes = []
    vtk_pc = vis.VtkPointCloud(pc_upright_camera)
    vtk_actors.append(vtk_pc.vtk_actor)

    # Visualize GT 3D boxes
    for box3d in gt_box3d_list:
        vtk_box3D = vis.vtk_box_3D(box3d)
        vtk_gt_boxes.append(vtk_box3D)

    # Visualize GT 2D boxes
    img_filename = os.path.join(IMG_DIR, '%06d.jpg' % file_num)
    vis.display_image(img_filename, gt_box2Ds=gt_box2d_list, gt_col=vis.Color.LightGreen)    

    print('\nTotal number of classes: %d' % len(objects))
    print('Image Size             : %s' % str(img.size))
    print('Point Cloud Size       : %d' % len(pc_upright_camera))

    return vtk_actors, { 'g': vtk_gt_boxes }

def vis_fpc(file_num):
    # Get dataset information
    calib, objects, pc_upright_camera, img, pc_image_coord = load_data(dataset, file_num)
    
    # Visualize point cloud
    vtk_actors = []
    vtk_gt_boxes = []
    vtk_pc = vis.VtkPointCloud(pc_upright_camera)
    vis.vtk_transform_actor(vtk_pc.vtk_actor, translate=(0,SEP*-2,0))
    vtk_actors.append(vtk_pc.vtk_actor)

    gt_box2d_list = []
    for idx in range(len(objects)):
        row, col = idx // COLS, idx % COLS
        obj = objects[idx]
        if obj.classname not in WHITE_LIST: continue
        gt_box2d_list.append(obj.box2d)

        # Extract frustum point cloud
        box2d, box3d_pts_3d, cropped_image, pc_in_box_fov, label, box3d_size, frustum_angle, img_dims = \
            process_object(calib, obj, pc_upright_camera, img, pc_image_coord, perturb_box2d=False)

        # Visualize frustum point cloud
        vtk_pc = vis.VtkPointCloud(pc_in_box_fov, gt_points=label, pred_points=[])
        vis.vtk_transform_actor(vtk_pc.vtk_actor, translate=(SEP*col,SEP*row,0))
        vtk_actors.append(vtk_pc.vtk_actor)

        # Visualize GT 3D boxes
        vtk_box_3D_full = vis.vtk_box_3D(box3d_pts_3d)
        vis.vtk_transform_actor(vtk_box_3D_full, translate=(0,SEP*-2,0))
        vtk_gt_boxes.append(vtk_box_3D_full)

        vtk_box3D = vis.vtk_box_3D(box3d_pts_3d)
        vis.vtk_transform_actor(vtk_box3D, translate=(SEP*col,SEP*row,0))
        vtk_gt_boxes.append(vtk_box3D)

    # Visualize GT 2D boxes
    img_filename = os.path.join(IMG_DIR, '%06d.jpg' % file_num)
    vis.display_image(img_filename, gt_box2Ds=gt_box2d_list, gt_col=vis.Color.LightGreen) 

    return vtk_actors, { 'g': vtk_gt_boxes }

def vis_seg_box_dataset(filename, number):
    rotate_to_center = True
    segbox_dataset = ROISegBoxDataset(WHITE_LIST, npoints=2048, split='training', 
        rotate_to_center=rotate_to_center, random_flip=False, random_shift=False, 
        overwritten_data_path=filename, from_rgb_detection=FLAGS.rgb_detection)

    classes, file_nums = [], []
    vtk_pcs_wo_col, vtk_pcs_w_col, vtk_gt_boxes, vtk_imgs, vtk_texts = [], [], [], [], []
    #for idx in range(number):
    choices = np.random.choice(len(segbox_dataset), number)
    choices.sort()
    for idx, choice in enumerate(choices):
        row, col = idx // COLS, idx % COLS
        if FLAGS.rgb_detection:
            point_set, img, rot, prob = segbox_dataset[choice]
        else:
            point_set, img, seg_gt, center, angle_class, angle_res, size_cls, size_res, box2d, rtilt, \
                k, rot, img_dims = segbox_dataset[choice]
        file_num = segbox_dataset.idx_l[choice]

        # Visualize point cloud
        vtk_pc_wo_col = vis.VtkPointCloud(point_set)
        vtk_pc_w_col = vis.VtkPointCloud(point_set, gt_points=seg_gt)
        vis.vtk_transform_actor(vtk_pc_wo_col.vtk_actor, translate=(SEP*col,SEP*row,0))
        vis.vtk_transform_actor(vtk_pc_w_col.vtk_actor, translate=(SEP*col,SEP*row,0))
        vtk_pcs_wo_col.append(vtk_pc_wo_col.vtk_actor)
        vtk_pcs_w_col.append(vtk_pc_w_col.vtk_actor)

        # Visualize GT 3D box
        if not FLAGS.rgb_detection:
            lwh = class2size(size_cls, size_res)
            ry  = class2angle(angle_class, angle_res, 12)
            box3d = get_3d_box(lwh, ry, center)
            vtk_box3D = vis.vtk_box_3D(box3d, color=vis.Color.LightGreen)
            vis.vtk_transform_actor(vtk_box3D, translate=(SEP*col,SEP*row,0))
            vtk_gt_boxes.append(vtk_box3D)

            calib = dataset.get_calibration(file_num)
            box3d_rot = rotate_pc_along_y(box3d.copy(), -rot)
            box3d_pts_3d_orig = calib.project_upright_camera_to_upright_depth(box3d_rot)
            box3d_pts_2d, _ = calib.project_upright_depth_to_image(box3d_pts_3d_orig)

        # Visualize Images
        box2d = segbox_dataset.box2d_l[choice]
        box2d_col = vis.Color.LightGreen if not FLAGS.rgb_detection else vis.Color.Orange
        img_filename = os.path.join(IMG_DIR, '%06d.jpg' % file_num)
        vtk_img = vis.vtk_image(img_filename, 
                                box2Ds_list=[[box2d]], box2Ds_cols=[box2d_col],
                                proj_box3Ds_list=[box3d_pts_2d], 
                                proj_box3Ds_cols=[vis.Color.LightGreen])
        vis.vtk_transform_actor(vtk_img, scale=(IMG_SCALE,IMG_SCALE,IMG_SCALE), 
                                rot=(0,180,180), translate=(-2+SEP*col,2+SEP*row,10))
        vtk_imgs.append(vtk_img)

        # Text information
        classes.append(segbox_dataset.cls_type_l[choice])
        file_nums.append(str(file_num))

    # Visualize text information
    vtk_texts.extend(vis.vtk_text(['Class:'] * len(classes), arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-1.5,3,2)))
    vtk_texts.extend(vis.vtk_text(classes, arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(0.5,3,2)))
    vtk_texts.extend(vis.vtk_text(['File:'] * len(file_nums), arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-1.5,3.5,2)))
    vtk_texts.extend(vis.vtk_text(file_nums, arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(0.25,3.5,2)))

    return vtk_pcs_w_col, { 'g': vtk_gt_boxes, 'c': vtk_pcs_wo_col, 'i': vtk_imgs, 't': vtk_texts }

def vis_box_pc_dataset(filename, number):
    boxpc_dataset = BoxPCFitDataset(classes=WHITE_LIST, 
                    center_perturbation=0.8, size_perturbation=0.2, angle_perturbation=np.pi,
                    npoints=2048, rotate_to_center=True, random_flip=True, random_shift=True, 
                    overwritten_data_path=filename)
    b_data, b_image, b_label, b_center, b_heading_class, b_heading_residual, b_size_class, \
    b_size_residual, b_box2d, b_rilts, b_ks, b_rot_angle, b_img_dims, b_one_hot_vec, \
    b_new_center, b_new_heading_class, b_new_heading_residual, b_new_size_class, \
    b_new_size_residual, b_box_iou, b_y_center_delta, b_y_size_delta, b_y_angle_delta = \
        boxpc_dataset.sample_batch(bsize=number, num_point=2048, num_channel=6, 
                                   proportion_of_boxpc_fit=0.5, 
                                   boxpc_nofit_bounds=[0.01,0.25], 
                                   boxpc_fit_bounds=[0.7,1.0], equal_samples_per_class=True)

    classes, box_ious = [], []
    vtk_pcs_with_col, vtk_gt_boxes, vtk_perb_boxes, vtk_imgs, vtk_pcs_wo_col, vtk_texts  = [], [], [], [], [], []
    for idx in range(len(b_data)):
        row, col = idx // COLS, idx % COLS
        cls_type = np.argmax(b_one_hot_vec[idx])
        point_set = b_data[idx]
        seg_gt = b_label[idx]
        iou3d = b_box_iou[idx]
        # GT 3D box
        center = b_center[idx]
        angle_class = b_heading_class[idx]
        angle_res = b_heading_residual[idx]
        size_cls = b_size_class[idx]
        size_res = b_size_residual[idx]
        # Perturbed 3D box
        new_center = b_new_center[idx]
        new_angle_class = b_new_heading_class[idx]
        new_angle_res = b_new_heading_residual[idx]
        new_size_cls = b_new_size_class[idx]
        new_size_res = b_new_size_residual[idx]

        # Visualize point cloud (with and without color)
        vtk_pc_wo_col = vis.VtkPointCloud(point_set)
        vtk_pc = vis.VtkPointCloud(point_set, gt_points=seg_gt)
        vis.vtk_transform_actor(vtk_pc_wo_col.vtk_actor, translate=(SEP*col,SEP*row,0))
        vis.vtk_transform_actor(vtk_pc.vtk_actor, translate=(SEP*col,SEP*row,0))
        vtk_pcs_wo_col.append(vtk_pc_wo_col.vtk_actor)
        vtk_pcs_with_col.append(vtk_pc.vtk_actor)

        # Visualize GT 3D box
        lwh = class2size(size_cls, size_res)
        ry  = class2angle(angle_class, angle_res, 12)
        gt_box3d = get_3d_box(lwh, ry, center)
        vtk_gt_box3D = vis.vtk_box_3D(gt_box3d, color=vis.Color.LightGreen)
        vis.vtk_transform_actor(vtk_gt_box3D, translate=(SEP*col,SEP*row,0))
        vtk_gt_boxes.append(vtk_gt_box3D)

        # Visualize Perb 3D box
        new_lwh = class2size(new_size_cls, new_size_res)
        new_ry  = class2angle(new_angle_class, new_angle_res, 12)
        perb_box3d = get_3d_box(new_lwh, new_ry, new_center)
        vtk_perb_box3D = vis.vtk_box_3D(perb_box3d, color=vis.Color.Blue)
        vis.vtk_transform_actor(vtk_perb_box3D, translate=(SEP*col,SEP*row,0))
        vtk_perb_boxes.append(vtk_perb_box3D)

        # Other information
        classes.append(class2type[cls_type].capitalize())
        box_iou3d, _ = box3d_iou(gt_box3d, perb_box3d)
        print('%d: %.3f, %.3f' % (idx, iou3d, box_iou3d - iou3d))
        box_ious.append(iou3d)

    # Visualize text information
    vtk_texts.extend(vis.vtk_text(['Class:'] * len(classes), arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-1.5,3,2)))
    vtk_texts.extend(vis.vtk_text(classes, arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(0.5,3,2)))
    vtk_texts.extend(vis.vtk_text(['Box:'] * len(box_ious), arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-1.5,4,2)))
    vtk_texts.extend(vis.vtk_text(box_ious, arr_type='float', color=True, sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(0,4,2)))

    key_to_actors_to_hide = { 'g': vtk_gt_boxes, 'p': vtk_perb_boxes, 'i': vtk_imgs, 'c': vtk_pcs_wo_col, 't': vtk_texts }

    return vtk_pcs_with_col, key_to_actors_to_hide

def vis_predictions2D(pred_folder, number_to_show=10):
    pred_files = os.listdir(pred_folder)
    valid_files = [int(line.rstrip()) for line in open(os.path.join(SUNRGBD_DATASET_DIR,'training','val_data_idx.txt'))]
    choices = np.random.choice(len(valid_files), number_to_show)
    
    file_nums = []
    vtk_imgs, vtk_texts = [], []
    for idx, i in enumerate(choices):
        row, col = idx // COLS, idx % COLS
        file_num = valid_files[i]

        # Get GT 2D boxes
        gt_box2d_list = []
        objects = dataset.get_label_objects(file_num)
        for i in range(len(objects)):
            obj = objects[i]
            if obj.classname not in WHITE_LIST: continue
            gt_box2d_list.append(obj.box2d)

        # Get Pred 2D boxes
        pred_box2d_list = []
        pred_file_path = os.path.join(pred_folder, '%06d.txt' % file_num)
        print(pred_file_path)
        if os.path.exists(pred_file_path):
            for line in open(pred_file_path, 'r'):
                t = line.rstrip().split(' ')
                pred_box2d_list.append(np.array([float(t[i]) for i in range(4,8)]))

        # Visualize Images
        img_filename = os.path.join(IMG_DIR, '%06d.jpg' % file_num)
        vtk_img = vis.vtk_image(img_filename, box2Ds_list=[pred_box2d_list], 
                                box2Ds_cols=[vis.Color.Red])
        vis.vtk_transform_actor(vtk_img, scale=(IMG_SCALE,IMG_SCALE,IMG_SCALE), 
                                rot=(0,180,180), translate=(-2+SEP*col,2+SEP*row,10))
        vtk_imgs.append(vtk_img)

        # Other information
        file_nums.append(str(file_num))

    # Visualize text information
    vtk_texts.extend(vis.vtk_text(['File:'] * len(file_nums), arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-1.5,3.5,2)))
    vtk_texts.extend(vis.vtk_text(file_nums, arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(0.25,3.5,2)))

    key_to_actors_to_hide = { 'i': vtk_imgs, 't': vtk_texts }
    return [], key_to_actors_to_hide

def vis_predictions3D(pred_files, gt_file, number_to_show=10, filenums=None):
    
    from roi_seg_box3d_dataset import class2type
    
    idx = 0
    COLS = number_to_show
    ap_infos = {}
    classes, file_nums, mean_box_ious, mean_seg_ious, box_ious, seg_ious = [], [], [], [], [], []
    vtk_pcs_with_col, vtk_pcs_wo_col, vtk_imgs, vtk_gt_boxes, vtk_pred_boxes, vtk_texts = [], [], [], [], [], []
    choices = []

    test_dataset = ROISegBoxDataset(WHITE_LIST, npoints=2048, 
                                    split='val', rotate_to_center=True, 
                                    overwritten_data_path=gt_file, 
                                    from_rgb_detection=False)

    for n, pred_file in enumerate(pred_files):
        # Lists of different items from predictions
        predictions = load_zipped_pickle(pred_file)
        ps_l, seg_gt_l, seg_pred_l, center_l, heading_cls_l, heading_res_l, size_cls_l, size_res_l, rot_angle_l, \
            score_l, cls_type_l, file_num_l, box2d_l, box3d_l = predictions
        if n == 0:
            # Choosing equal number of objects per class to display
            cls_types = []
            options = {}
            for i, cls_type in enumerate(cls_type_l):
                if not class2type[cls_type] in WHITE_LIST: continue
                if options.get(cls_type) is None:
                    options[cls_type] = [i]
                    cls_types.append(cls_type)
                else:
                    options[cls_type].append(i)

            # Make use of array_split to divide into fairly equal groups
            arr = np.array_split([1] * number_to_show, len(options.keys()))
            random.shuffle(arr)
            for i, group in enumerate(arr):
                cls_type = cls_types[i]
                choice_list = np.random.choice(options[cls_type], len(group), replace=False) #replace=True)
                choices.extend(choice_list)
            print('Number of objects in whitelist: %d' % len(options))

        # Compute overall statistics
        if not FLAGS.rgb_detection:
            print('==== Computing overall statistics for %s ====' % pred_file)
            from evaluate import evaluate_predictions, get_ap_info
            rec, prec, ap, mean_ap = evaluate_predictions(predictions, dataset, CLASSES, 
                                                          test_dataset, WHITE_LIST)
            ap['Mean AP'] = mean_ap
            for classname in ap.keys():
                if ap_infos.get(classname) is None: ap_infos[classname] = []
                ap_infos[classname].append('%11s: [%.1f]' % (classname, 100. * ap[classname]))

            box_iou_sum, seg_iou_sum = 0, 0
            for i in range(len(ps_l)):
                seg_gt = seg_gt_l[i]
                box3d = box3d_l[i]
                seg_pred = seg_pred_l[i]
                center = center_l[i]
                heading_cls = heading_cls_l[i]
                heading_res = heading_res_l[i]
                size_cls = size_cls_l[i]
                size_res = size_res_l[i]
                rot_angle = rot_angle_l[i]

                gt_box3d = rotate_pc_along_y(np.copy(box3d), rot_angle)
                heading_angle = class2angle(heading_cls, heading_res, NUM_HEADING_BIN)
                box_size = class2size(size_cls, size_res) 
                pred_box3d = get_3d_box(box_size, heading_angle, center)

                # Box IOU
                shift_arr = np.array([4,5,6,7,0,1,2,3])
                box_iou3d, _ = box3d_iou(gt_box3d[shift_arr,:], pred_box3d)
                # Seg IOU
                seg_iou = get_seg_iou(seg_gt, seg_pred, 2)

                box_iou_sum += box_iou3d
                seg_iou_sum += seg_iou
            mean_box_iou = box_iou_sum / len(ps_l)
            mean_seg_iou = seg_iou_sum / len(ps_l)
            mean_box_ious.append(mean_box_iou)
            mean_seg_ious.append(mean_seg_iou)
             
        for i in choices:
            row, col = idx // COLS, idx % COLS
            idx += 1
            ps = ps_l[i]
            seg_pred = seg_pred_l[i]
            center = center_l[i]
            heading_cls = heading_cls_l[i]
            heading_res = heading_res_l[i]
            size_cls = size_cls_l[i]
            size_res = size_res_l[i]
            rot_angle = rot_angle_l[i]
            score = score_l[i]
            cls_type = cls_type_l[i]
            file_num = file_num_l[i]
            seg_gt = seg_gt_l[i] if not FLAGS.rgb_detection else []
            box2d = box2d_l[i]
            box3d = box3d_l[i] if not FLAGS.rgb_detection else []

            # Visualize point cloud (with and without color)
            vtk_pc_wo_col = vis.VtkPointCloud(ps)
            vtk_pc = vis.VtkPointCloud(ps, gt_points=seg_gt, pred_points=seg_pred)
            vis.vtk_transform_actor(vtk_pc_wo_col.vtk_actor, translate=(SEP*col,SEP*row,0))
            vis.vtk_transform_actor(vtk_pc.vtk_actor, translate=(SEP*col,SEP*row,0))
            vtk_pcs_wo_col.append(vtk_pc_wo_col.vtk_actor)
            vtk_pcs_with_col.append(vtk_pc.vtk_actor)

            # Visualize GT 3D box
            if FLAGS.rgb_detection:
                objects = dataset.get_label_objects(file_num)
                calib = dataset.get_calibration(file_num)
                for obj in objects:
                    if obj.classname not in WHITE_LIST: continue
                    box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib)
                    box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
                    box3d_pts_3d = rotate_pc_along_y(np.copy(box3d_pts_3d), rot_angle)
                    vtk_box3D = vis.vtk_box_3D(box3d_pts_3d, color=vis.Color.LightGreen)
                    vis.vtk_transform_actor(vtk_box3D, translate=(SEP*col,SEP*row,0))
                    vtk_gt_boxes.append(vtk_box3D)
            else:
                gt_box3d = rotate_pc_along_y(np.copy(box3d), rot_angle)
                vtk_gt_box3D = vis.vtk_box_3D(gt_box3d, color=vis.Color.LightGreen)
                vis.vtk_transform_actor(vtk_gt_box3D, translate=(SEP*col,SEP*row,0))
                vtk_gt_boxes.append(vtk_gt_box3D)

            # Visualize Pred 3D box
            heading_angle = class2angle(heading_cls, heading_res, NUM_HEADING_BIN)
            box_size = class2size(size_cls, size_res) 
            pred_box3d = get_3d_box(box_size, heading_angle, center)
            vtk_pred_box3D = vis.vtk_box_3D(pred_box3d, color=vis.Color.White)
            vis.vtk_transform_actor(vtk_pred_box3D, translate=(SEP*col,SEP*row,0))
            vtk_pred_boxes.append(vtk_pred_box3D)

            # Visualize Images
            box2d_col = vis.Color.LightGreen if not FLAGS.rgb_detection else vis.Color.Orange
            img_filename = os.path.join(IMG_DIR, '%06d.jpg' % file_num)
            vtk_img = vis.vtk_image(img_filename, box2Ds_list=[[box2d]], box2Ds_cols=[box2d_col])
            vis.vtk_transform_actor(vtk_img, scale=(IMG_SCALE,IMG_SCALE,IMG_SCALE), 
                                    rot=(0,180,180), translate=(-2+SEP*col,2+SEP*row,10))
            vtk_imgs.append(vtk_img)

            # Other information
            classes.append(class2type[cls_type].capitalize())
            file_nums.append(str(file_num))
            if not FLAGS.rgb_detection:
                shift_arr = np.array([4,5,6,7,0,1,2,3])
                box_iou3d, _ = box3d_iou(gt_box3d[shift_arr,:], pred_box3d)
                box_ious.append(box_iou3d)
                seg_iou = get_seg_iou(seg_gt, seg_pred, 2)
                seg_ious.append(seg_iou)

    # Visualize overall statistics
    vtk_texts.extend(vis.vtk_text([('Model: %s' % pred_file.split('/')[-1]) for pred_file in pred_files], arr_type='text', sep=SEP, cols=1, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-14.5,2.5,2)))
    vtk_texts.extend(vis.vtk_text(['Mean Box IOU:'] * len(pred_files), arr_type='text', sep=SEP, cols=1, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-14.5,3,2)))
    vtk_texts.extend(vis.vtk_text(mean_box_ious, arr_type='float', color=True, sep=SEP, cols=1, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-10,3,2)))
    vtk_texts.extend(vis.vtk_text(['Mean Seg IOU:'] * len(pred_files), arr_type='text', sep=SEP, cols=1, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-14.5,3.5,2)))
    vtk_texts.extend(vis.vtk_text(mean_seg_ious, arr_type='float', color=True, sep=SEP, cols=1, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-10,3.5,2)))
    for i, (cls_name, ap_info) in enumerate(ap_infos.items()):
        vtk_texts.extend(vis.vtk_text(ap_info, arr_type='text', color=True, sep=SEP, cols=1, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-14.5,4+i*0.5,2)))

    # Visualize text information
    vtk_texts.extend(vis.vtk_text(['Class:'] * len(classes), arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-1.5,3,2)))
    vtk_texts.extend(vis.vtk_text(classes, arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(0.5,3,2)))
    vtk_texts.extend(vis.vtk_text(['File:'] * len(file_nums), arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-1.5,3.5,2)))
    vtk_texts.extend(vis.vtk_text(file_nums, arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(0.25,3.5,2)))
    if not FLAGS.rgb_detection:
        vtk_texts.extend(vis.vtk_text(['Box:'] * len(box_ious), arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-1.5,4,2)))
        vtk_texts.extend(vis.vtk_text(box_ious, arr_type='float', color=True, sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(0,4,2)))
        vtk_texts.extend(vis.vtk_text(['Seg:'] * len(seg_ious), arr_type='text', sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(-1.5,4.5,2)))
        vtk_texts.extend(vis.vtk_text(seg_ious, arr_type='float', color=True, sep=SEP, cols=COLS, scale=TEXT_SCALE, rot=TEXT_ROT, translate=(0,4.5,2)))

    key_to_actors_to_hide = { 'g': vtk_gt_boxes, 'p': vtk_pred_boxes, 'i': vtk_imgs, 'c': vtk_pcs_wo_col, 't': vtk_texts }
    return vtk_pcs_with_col, key_to_actors_to_hide

# 3D BOX: Get pts velo in 3d box
def get_2d_box(pc_image, h, w):
    x_min = max(np.min(pc_image[:,0]), 0)
    x_max = min(np.max(pc_image[:,0]), w)
    y_min = max(np.min(pc_image[:,1]), 0)
    y_max = min(np.max(pc_image[:,1]), h)
    assert(x_max > x_min)
    assert(y_max > y_min)
    return [x_min,y_min,x_max,y_max]

def get_seg_iou(seg_gt, seg_pred, num_seg_classes):
    class_ious = []
    for l in range(num_seg_classes):
        if (np.sum(seg_gt==l) == 0) and (np.sum(seg_pred==l) == 0): 
            # class is not present, no logitsiction as well
            assert(False)
            class_ious.append(1.0)
        else:
            class_ious.append(np.sum((seg_gt==l) & (seg_pred==l)) / float(np.sum((seg_gt==l) | (seg_pred==l))))
    return np.mean(class_ious)

if __name__ == '__main__':
  
    # Different visualizations
    if FLAGS.vis == 'pc':
        if FLAGS.filenum is None:
            raise Exception('File number is not specified.')
        vtk_actors, key_to_actors_to_hide = vis_point_cloud(FLAGS.filenum)

    elif FLAGS.vis == 'fpc':
        if FLAGS.filenum is None:
            raise Exception('File number is not specified.')
        vtk_actors, key_to_actors_to_hide = vis_fpc(FLAGS.filenum)

    elif FLAGS.vis == 'seg_box':
        if FLAGS.filename is None:
            raise Exception('Filename is not specified.')
        vtk_actors, key_to_actors_to_hide = vis_seg_box_dataset(FLAGS.filename, FLAGS.num)

    elif FLAGS.vis == 'box_pc':
        if FLAGS.filename is None:
            raise Exception('File is not specified.')
        vtk_actors, key_to_actors_to_hide = vis_box_pc_dataset(FLAGS.filename, FLAGS.num)

    elif FLAGS.vis == 'pred2d':
        if FLAGS.pred_files is None:
            raise Exception('Prediction files are not specified.')
        vtk_actors, key_to_actors_to_hide = vis_predictions2D(FLAGS.pred_files[0], FLAGS.num)

    elif FLAGS.vis == 'pred3d':
        if FLAGS.pred_files is None:
            raise Exception('Prediction files are not specified.')
        if FLAGS.gt_file is None:
            raise Exception('Ground truth file is not specified.')
        vtk_actors, key_to_actors_to_hide = vis_predictions3D(FLAGS.pred_files, FLAGS.gt_file,
                                                              FLAGS.num, filenums=FLAGS.filenums)

    # Render
    vis.start_render(vtk_actors, key_to_actors_to_hide=key_to_actors_to_hide, background_col=vis.Color.Black)