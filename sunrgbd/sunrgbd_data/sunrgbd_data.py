
''' Helper class and functions for loading SUN RGB-D objects

Author: Charles R. Qi
Date: October 2017
'''

import os
import sys
import argparse
from os.path import join as pjoin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import utils
import numpy as np
from PIL import Image
from utils import random_shift_box2d, extract_pc_in_box3d

data_dir = BASE_DIR
SUNRGBD_DATASET_DIR = '/home/yewsiang/Transferable3D/dataset/mysunrgbd'

class sunrgbd_object(object):
    ''' Load and parse object data '''
    def __init__(self, root_dir, split='training'):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        # TODO(YS): Current num samples are different
        if split == 'training':
            self.num_samples = 10335
        elif split == 'testing':
            self.num_samples = 2860
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.depth_dir = os.path.join(self.split_dir, 'depth')
        self.label_dir = os.path.join(self.split_dir, 'label_dimension')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        return utils.load_image(img_filename)

    def get_depth(self, idx): 
        depth_filename = os.path.join(self.depth_dir, '%06d.txt'%(idx))
        return utils.load_depth_points(depth_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.SUNRGBD_Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert(self.split=='training') 
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return utils.read_sunrgbd_label(label_filename)

def load_data(dataset, data_idx):
    calib = dataset.get_calibration(data_idx)
    objects = dataset.get_label_objects(data_idx)
    pc_upright_depth = dataset.get_depth(data_idx)
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    pc_upright_camera[:,0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:,0:3])
    pc_upright_camera[:,3:] = pc_upright_depth[:,3:]
    img = dataset.get_image(data_idx)
    img_height, img_width, img_channel = img.shape
    pc_image_coord,_ = calib.project_upright_depth_to_image(pc_upright_depth)
    return calib, objects, pc_upright_camera, img, pc_image_coord

def process_object(calib, obj, pc_upright_camera, img, pc_image_coord, perturb_box2d=False, num_points=2048):
    # Augment data by box2d perturbation
    box2d = obj.box2d
    if perturb_box2d:
        xmin,ymin,xmax,ymax = random_shift_box2d(box2d)
        # print(xmin,ymin,xmax,ymax)
    else:
        xmin,ymin,xmax,ymax = box2d
    box2d = np.array([xmin,ymin,xmax,ymax])
    box_fov_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)
    pc_in_box_fov = pc_upright_camera[box_fov_inds,:]

    # Get frustum angle (according to center pixel in 2D BOX)
    box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
    uvdepth = np.zeros((1,3))
    uvdepth[0,0:2] = box2d_center
    uvdepth[0,2] = 20 # some random depth
    box2d_center_upright_camera = calib.project_image_to_upright_camerea(uvdepth)
    #print('UVdepth, center in upright camera: ', uvdepth, box2d_center_upright_camera)
    frustum_angle = -1 * np.arctan2(box2d_center_upright_camera[0,2], box2d_center_upright_camera[0,0]) # angle as to positive x-axis as in the Zoox paper
    #print('Frustum angle: ', frustum_angle)

    # 3D BOX: Get pts velo in 3d box
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib) 
    box3d_pts_3d = calib.project_upright_depth_to_upright_camera(box3d_pts_3d)
    _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
    # print(len(inds))
    label = np.zeros((pc_in_box_fov.shape[0]))
    label[inds] = 1

    # Get cropped image
    cropped_image = img[int(ymin):int(ymax), int(xmin):int(xmax),:]
    h, w, _ = img.shape
    img_dims = [h, w]

    # Get 3D BOX heading
    # print('Orientation: ', obj.orientation)
    # print('Heading angle: ', obj.heading_angle)

    # Get 3D BOX size
    box3d_size = np.array([2*obj.l,2*obj.w,2*obj.h])
    # print('Box3d size: ', box3d_size)
    # print('Type: ', obj.classname)
    # print('Num of point: ', pc_in_box_fov.shape[0])

    # Subsample points..
    num_point = pc_in_box_fov.shape[0]
    if num_point > num_points:
        choice = np.random.choice(pc_in_box_fov.shape[0], num_points, replace=False)
        pc_in_box_fov = pc_in_box_fov[choice,:]
        label = label[choice]
    
    return box2d, box3d_pts_3d, cropped_image, pc_in_box_fov, label, box3d_size, frustum_angle, img_dims

def extract_roi_seg(idx_filename, split, output_filename, viz, perturb_box2d=False, augmentX=1, type_whitelist=['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']):
    
    print('Extracting roi_seg from: %s' % idx_filename)
    dataset = sunrgbd_object(SUNRGBD_DATASET_DIR, split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    box3d_list = [] # (8,3) array in upright depth coord
    image_list = [] # (h_i, w_i, 3) - different height and width for each image
    input_list = [] # channel number = 6, xyz,rgb in upright depth coord
    label_list = [] # 1 for roi object, 0 for clutter
    type_list = [] # string e.g. bed
    heading_list = [] # face of object angle, radius of clockwise angle from positive x axis in upright camera coord
    box3d_size_list = [] # array of l,w,h
    rtilts_list = [] # (3,3) array used for projection onto image
    ks_list = [] # (3,3) array used for projection onto image
    frustum_angle_list = [] # angle of 2d box center from pos x-axis (clockwise)
    img_dims_list = [] # dimensions of the full images (not just the object)

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib, objects, pc_upright_camera, img, pc_image_coord = load_data(dataset, data_idx)
        
        #print('PC image coord: ', pc_image_coord)

        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.classname not in type_whitelist: continue

            # 2D BOX: Get pts rect backprojected 
            for _ in range(augmentX):
                #try:
                box2d, box3d_pts_3d, cropped_image, pc_in_box_fov, label, box3d_size, frustum_angle, img_dims = \
                    process_object(calib, obj, pc_upright_camera, img, pc_image_coord, perturb_box2d=perturb_box2d)

                # Reject object with too few points
                if np.sum(label) < 5:
                    continue

                id_list.append(data_idx)
                box2d_list.append(box2d)
                box3d_list.append(box3d_pts_3d)
                image_list.append(cropped_image)
                input_list.append(pc_in_box_fov)
                label_list.append(label)
                type_list.append(obj.classname)
                heading_list.append(obj.heading_angle)
                box3d_size_list.append(box3d_size)
                rtilts_list.append(calib.Rtilt)
                ks_list.append(calib.K)
                frustum_angle_list.append(frustum_angle)
                img_dims_list.append(img_dims)

                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

    print('Average pos ratio: ', pos_cnt/float(all_cnt))
    print('Average npoints: ', float(all_cnt)/len(id_list))

    utils.save_zipped_pickle([id_list, box2d_list, box3d_list, image_list, input_list, label_list, \
                              type_list, heading_list, box3d_size_list, rtilts_list, ks_list, \
                              frustum_angle_list, img_dims_list], output_filename)

def get_box3d_dim_statistics(idx_filename, type_whitelist=['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']):
    dataset = sunrgbd_object(SUNRGBD_DATASET_DIR)
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.classname not in type_whitelist: continue
            heading_angle = -1 * np.arctan2(obj.orientation[1], obj.orientation[0])
            dimension_list.append(np.array([obj.l,obj.w,obj.h])) 
            type_list.append(obj.classname) 
            ry_list.append(heading_angle)

    import cPickle as pickle
    with open('box3d_dimensions.pickle','wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)

def read_det_folder(det_folder):
    filenames = os.listdir(det_folder)
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for filename in filenames:
        img_id = int(filename[0:6])
        full_filename = os.path.join(det_folder, filename)
        for line in open(full_filename, 'r'):
            t = line.rstrip().split(" ")
            prob = float(t[-1])
            # (YS) There are many values that are negative
            #if prob < 0.05: continue 
            id_list.append(img_id)
            type_list.append(t[0]) 
            prob_list.append(prob)
            box2d_list.append(np.array([float(t[i]) for i in range(4,8)]))
    return id_list, type_list, box2d_list, prob_list


def extract_roi_seg_from_rgb_detection(det_folder, split, output_filename, viz, valid_id_list=None, type_whitelist=['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']):
    ''' Extract data pairs for RoI point set segmentation from RGB detector outputed 2D boxes.
        
        Input:
            det_folder: contains files for each frame, lines in each file are type -1 -10 -10 xmin ymin xmax ymax ... prob
            split: corresponding to official either trianing or testing
            output_filename: the name for output .pickle file
            valid_id_list: specify a list of valid image IDs
        Output:
            None (will write a .pickle file to the disk)

        Usage: extract_roi_seg_from_rgb_detection("val_result_folder", "training", "roi_seg_val_rgb_detector_0908.pickle")

    '''
    dataset = sunrgbd_object(SUNRGBD_DATASET_DIR, split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_folder(det_folder)
    cache_id = -1
    cache = None
    
    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    image_list = [] # (h_i, w_i, 3) - different height and width for each image
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    for det_idx in range(len(det_id_list)):
        data_idx = det_id_list[det_idx]
        if valid_id_list is not None and data_idx not in valid_id_list: continue
        print('det idx: %d/%d, data idx: %d' % (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx)
            pc_upright_depth = dataset.get_depth(data_idx)
            pc_upright_camera = np.zeros_like(pc_upright_depth)
            pc_upright_camera[:,0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:,0:3])
            pc_upright_camera[:,3:] = pc_upright_depth[:,3:]

            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            pc_image_coord,_ = calib.project_upright_depth_to_image(pc_upright_depth)
            cache = [calib,pc_upright_camera,pc_image_coord]
            cache_id = data_idx
        else:
            calib,pc_upright_camera,pc_image_coord = cache

       
        if det_type_list[det_idx] not in type_whitelist: continue

        # 2D BOX: Get pts rect backprojected 
        xmin,ymin,xmax,ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)
        pc_in_box_fov = pc_upright_camera[box_fov_inds,:]
        
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
        uvdepth = np.zeros((1,3))
        uvdepth[0,0:2] = box2d_center
        uvdepth[0,2] = 20 # some random depth
        box2d_center_upright_camera = calib.project_image_to_upright_camerea(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_upright_camera[0,2], box2d_center_upright_camera[0,0]) # angle as to positive x-axis as in the Zoox paper
        
        # Subsample points..
        num_point = pc_in_box_fov.shape[0]
        if num_point > 2048:
            choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace=False)
            pc_in_box_fov = pc_in_box_fov[choice,:]

        # Get cropped image
        cropped_image = img[int(ymin):int(ymax), int(xmin):int(xmax),:]
 
        # Pass objects that are too small
        if len(pc_in_box_fov)<5:
            continue
       
        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        image_list.append(cropped_image)
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)
    
    utils.save_zipped_pickle([id_list, box2d_list, image_list, input_list, type_list, \
                              frustum_angle_list, prob_list], output_filename)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--option', default='', choices=['stats', 'rgb_detection'], help='To visualize, to retrieve the statistics or to extract the rois with 2D predictions.')
    parser.add_argument('--test_data', default='', choices=['train', 'val', 'trainval', 'test'], help='Dataset to use for testing.')
    parser.add_argument('--rgb_detection_path', default=None, help='Path for the 2D detection results')
    parser.add_argument('--output_filename', default=None, help='Name for the output pickle filename')
    FLAGS = parser.parse_args()

    if FLAGS.option == 'stats':
        get_box3d_dim_statistics('/home/rqi/Data/mysunrgbd/training/train_data_idx.txt')

    elif FLAGS.option == 'rgb_detection':
        if FLAGS.rgb_detection_path is None:
            raise Exception('Please provide an 2D detection path.')
        if FLAGS.output_filename is None:
            # raise Exception('Please provide a name for the output pickle filename.')
            FLAGS.output_filename = FLAGS.rgb_detection_path.split('/')[-1]
            print(FLAGS.output_filename)
        assert(FLAGS.test_data in ['train', 'val', 'trainval', 'test'])
        valid_id_list = [int(line.rstrip()) for line in open(pjoin(SUNRGBD_DATASET_DIR,'training','%s_data_idx.txt' % FLAGS.test_data))]
        extract_roi_seg_from_rgb_detection(FLAGS.rgb_detection_path, 'training', 
            output_filename=pjoin(BASE_DIR,'..','frustums','%s_%s.zip.pickle' % (FLAGS.test_data, FLAGS.output_filename)), 
            valid_id_list=valid_id_list, viz=False)

    else:
        # Train on train, Test on val
        extract_roi_seg(pjoin(SUNRGBD_DATASET_DIR,'training','train_mini_data_idx.txt'), 'training',  
            output_filename=pjoin(BASE_DIR,'..','frustums','train_mini.zip.pickle'), viz=False, augmentX=1)
        extract_roi_seg(pjoin(SUNRGBD_DATASET_DIR,'training','train_data_idx.txt'), 'training',  
            output_filename=pjoin(BASE_DIR,'..','frustums','train_aug5x.zip.pickle'), viz=False, augmentX=5) 
        extract_roi_seg(pjoin(SUNRGBD_DATASET_DIR,'training','val_data_idx.txt'), 'training',
            output_filename=pjoin(BASE_DIR,'..','frustums','val.zip.pickle'), viz=False, augmentX=1)

        # Train on trainval, Test on test 
        extract_roi_seg(pjoin(SUNRGBD_DATASET_DIR,'training','trainval_data_idx.txt'), 'training',
            output_filename=pjoin(BASE_DIR,'..','frustums','trainval_aug5x.zip.pickle'), viz=False, augmentX=5)
        extract_roi_seg(pjoin(SUNRGBD_DATASET_DIR,'training','test_data_idx.txt'), 'training',
            output_filename=pjoin(BASE_DIR,'..','frustums','test.zip.pickle'), viz=False, augmentX=1)