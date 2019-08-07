
""" Dataset for training of BoxPC models.

Author: Tang Yew Siang
Date: July 2019
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../sunrgbd_data'))
sys.path.append(os.path.join(BASE_DIR, '../../train'))

import random
import numpy as np
from collections import deque
from box_util import box3d_iou
from utils import roty, load_zipped_pickle
from roi_seg_box3d_dataset import get_3d_box, rotate_pc_along_y, angle2class, class2angle, size2class, class2size

type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
class2type = {type2class[t]:t for t in type2class}
type2onehotclass={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
type_mean_size = {'bathtub'    : np.array([0.765840,1.398258,0.472728]),
                  'bed'        : np.array([2.114256,1.620300,0.927272]),
                  'bookshelf'  : np.array([0.404671,1.071108,1.688889]),
                  'chair'      : np.array([0.591958,0.552978,0.827272]),
                  'desk'       : np.array([0.695190,1.346299,0.736364]),
                  'dresser'    : np.array([0.528526,1.002642,1.172878]),
                  'night_stand': np.array([0.500618,0.632163,0.683424]),
                  'sofa'       : np.array([0.923508,1.867419,0.845495]),
                  'table'      : np.array([0.791118,1.279516,0.718182]),
                  'toilet'     : np.array([0.699104,0.454178,0.756250])}
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 10
NUM_CLASS = 10

def get_box3d_iou(center_A, box_size_A, heading_angle_A, center_B, box_size_B, heading_angle_B):
    corners_3d_A = get_3d_box(box_size_A, heading_angle_A, center_A)
    corners_3d_B = get_3d_box(box_size_B, heading_angle_B, center_B)
    iou_3d, iou_2d = box3d_iou(corners_3d_A, corners_3d_B) 
    return iou_3d, iou_2d

def inrange(val, low, high):
    return (val > low) and (val < high)

class BoxPCFitDataset(object):
    def __init__(self, classes, npoints, center_perturbation, size_perturbation, angle_perturbation, 
        classes_to_drop=[], classes_to_drop_prob=0, random_flip=False, random_shift=False, rotate_to_center=False, overwritten_data_path=None):
        self.classes = classes
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.center_perturbation = center_perturbation
        self.size_perturbation = size_perturbation
        self.angle_perturbation = angle_perturbation
        self.prepared_batches = deque()
        assert(overwritten_data_path is not None)
        
        idx_l, box2d_l, box3d_l, image_crop_l, points_l, label_l, cls_type_l, heading_l, size_l, \
            rtilt_l, k_l, frustum_angle_l, img_dims_l = load_zipped_pickle(overwritten_data_path)

        # Maps cls_type to it's idx within (self.label_2Dl, self.heading_2Dl, etc) to allow class-wise sampling
        self.cls_to_idx_map = {}
        # Filtering out classes that we do not want
        self.idx_l, self.box2d_l, self.box3d_l, self.image_crop_l, self.points_l, self.label_l, \
            self.cls_type_l, self.heading_l, self.size_l, self.rtilt_l, self.k_l, \
            self.frustum_angle_l, self.img_dims_l = [], [], [], [], [], [], [], [], [], [], [], [], []

        np.random.seed(20)
        cls_idx = 0
        for idx, box2d, box3d, image_crop, points, label, cls_type, heading, size, rtilt, k, \
            frustum_angle, img_dims in zip(idx_l, box2d_l, box3d_l, image_crop_l, points_l, \
            label_l, cls_type_l, heading_l, size_l, rtilt_l, k_l, frustum_angle_l, img_dims_l):

            if cls_type in self.classes:

                if cls_type in classes_to_drop and (np.random.rand() < classes_to_drop_prob):
                    continue

                if self.cls_to_idx_map.get(cls_type) is None:
                    self.cls_to_idx_map[cls_type] = [cls_idx]
                else:
                    self.cls_to_idx_map[cls_type].append(cls_idx)
                cls_idx += 1

                self.idx_l.append(idx)
                self.box2d_l.append(box2d)
                self.box3d_l.append(box3d)
                #self.image_crop_l.append(image_crop)
                self.points_l.append(points)
                self.label_l.append(label)
                self.cls_type_l.append(cls_type)
                self.heading_l.append(heading)
                self.size_l.append(size)
                self.rtilt_l.append(rtilt)
                self.k_l.append(k)
                self.frustum_angle_l.append(frustum_angle)
                self.img_dims_l.append(img_dims)

    def __len__(self):
        return len(self.points_l)

    def get(self, index, is_boxpc_fit, boxpc_nofit_bounds, boxpc_fit_bounds):
        # ------------------------------ INPUTS ----------------------------
        # compute one hot vector
        cls_type = self.cls_type_l[index]
        assert(cls_type in self.classes)
        one_hot_vec = np.zeros((NUM_CLASS))
        one_hot_vec[type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.points_l[index]
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # Get image crop
        image = None
        # image = self.image_crop_l[index] 
        # image = cv2.resize(image, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR) 

        rot_angle = self.get_center_view_rot_angle(index)
        box2d = self.box2d_l[index]
        rtilt = self.rtilt_l[index]
        k = self.k_l[index]
        img_dims = self.img_dims_l[index]
        
        # ------------------------------ LABELS ----------------------------
        seg = self.label_l[index] 
        seg = seg[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_l[index] - rot_angle
        else:
            heading_angle = self.heading_l[index]

        # Size
        size = self.size_l[index]


        # Data Augmentation
        to_flip = (np.random.random() > 0.5)
        if self.random_flip and to_flip:
            point_set[:,0] *= -1
            box3d_center[0] *= -1
            heading_angle = np.pi - heading_angle

        if self.random_shift:
            dist  = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
            shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
            height_shift = np.random.random()*0.4-0.2 # randomly shift +-0.2 meters
            point_set[:,2] += shift
            point_set[:,1] += height_shift
            box3d_center[2] += shift
            box3d_center[1] += height_shift


        # Perturb points and box to different IOU
        boxpc_bounds = boxpc_fit_bounds if is_boxpc_fit else boxpc_nofit_bounds
        new_box3d_center, new_size, new_heading_angle, box_iou, y_center_delta, y_size_delta, \
        y_angle_delta = self.perturb_box_to_diff_ious(box3d_center, size, heading_angle, boxpc_bounds)


        size_class, size_residual = size2class(size, self.cls_type_l[index])
        angle_class, angle_residual = angle2class(heading_angle, NUM_HEADING_BIN)

        new_size_class, new_size_residual = size2class(new_size, self.cls_type_l[index])
        new_angle_class, new_angle_residual = angle2class(new_heading_angle, NUM_HEADING_BIN)

        return point_set, image, seg, box3d_center, angle_class, angle_residual, size_class, \
            size_residual, box2d, rtilt, k, rot_angle, img_dims, one_hot_vec, new_box3d_center, \
            new_angle_class, new_angle_residual, new_size_class, new_size_residual, box_iou, \
            y_center_delta, y_size_delta, y_angle_delta

    def get_center_view_rot_angle(self, index):
        return np.pi/2.0 + self.frustum_angle_l[index]

    def get_box3d_center(self, index):
        box3d_center = (self.box3d_l[index][0,:] + self.box3d_l[index][6,:])/2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        box3d_center = (self.box3d_l[index][0,:] + self.box3d_l[index][6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), self.get_center_view_rot_angle(index)).squeeze()
        
    def get_center_view_box3d(self, index):
        box3d = self.box3d_l[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Input ps is NxC points with first 3 channels as XYZ
            z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.points_l[index])
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle(index))

    def perturb_box_to_diff_ious(self, box3d_center, size, heading_angle, iou_bounds):

        # Less perturbation if iou_bounds close to 1
        iou_mean = np.mean(iou_bounds)
        center_perturbation = self.center_perturbation * (1 - iou_mean)
        size_perturbation   = self.size_perturbation * (1 - iou_mean)
        angle_perturbation  = self.angle_perturbation * (1 - iou_mean)
        
        # Keep perturbing until we have the desired IOU
        iou3d = -1
        count = 0
        while (not inrange(iou3d, iou_bounds[0], iou_bounds[1])):
            # Perturb Box center
            y_center_delta = np.random.uniform(-center_perturbation, center_perturbation, size=3)
            new_box3d_center = box3d_center + y_center_delta

            # Perturb Box size
            y_size_delta = np.multiply(size, np.random.uniform(-size_perturbation, +size_perturbation, size=3))
            new_size = size + y_size_delta

            # Always perturb heading angle
            y_angle_delta = np.random.uniform(0, angle_perturbation)
            new_heading_angle = heading_angle + y_angle_delta

            iou3d, _ = get_box3d_iou(box3d_center, size, heading_angle, \
                                     new_box3d_center, new_size, new_heading_angle)
            assert(iou3d >= 0. and iou3d <= 1.)

            #print(iou3d)
            count += 1

        #print('Number of counts to get desired IOU of [%.2f, %.2f]: %d' % (iou_bounds[0], iou_bounds[1], count))

        return new_box3d_center, new_size, new_heading_angle, iou3d, y_center_delta, y_size_delta, y_angle_delta

    def prepare_batches_for_one_epoch(self, sampling_method, num_batches, bsize, num_point, num_channel, equal_classes_prob, proportion_of_boxpc_fit, boxpc_nofit_bounds, boxpc_fit_bounds):
        assert(len(self.prepared_batches) == 0)

        train_idxs = np.arange(0, len(self))
        np.random.shuffle(train_idxs)
        for batch_idx in range(num_batches):

            if sampling_method == 'BATCH':
                start_idx = batch_idx * bsize
                end_idx   = (batch_idx+1) * bsize
                batch_data_tuple = self.get_batch(train_idxs, start_idx, end_idx, num_point, 
                    num_channel, proportion_of_boxpc_fit, boxpc_nofit_bounds, boxpc_fit_bounds)

            elif sampling_method == 'SAMPLE':
                sample_equal_class = (np.random.rand() < equal_classes_prob)
                batch_data_tuple = self.sample_batch(bsize, num_point, num_channel,
                    proportion_of_boxpc_fit, boxpc_nofit_bounds, boxpc_fit_bounds, 
                    equal_samples_per_class=sample_equal_class)

            self.prepared_batches.append(batch_data_tuple)

    def get_prepared_batch(self):
        return self.prepared_batches.pop()

    def get_prepared_batch_without_removing(self, index):
        return self.prepared_batches[index]

    def get_batch(self, idxs, start_idx, end_idx, num_point, num_channel, proportion_of_boxpc_fit, boxpc_nofit_bounds, boxpc_fit_bounds):
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, num_point, num_channel))
        #batch_image = np.zeros((bsize, IMG_H, IMG_W, IMG_CHANNELS))
        batch_label = np.zeros((bsize, num_point), dtype=np.int32)
        batch_center = np.zeros((bsize, 3))
        batch_heading_class = np.zeros((bsize,), dtype=np.int32)
        batch_heading_residual = np.zeros((bsize,))
        batch_size_class = np.zeros((bsize,), dtype=np.int32)
        batch_size_residual = np.zeros((bsize, 3))
        batch_box2d = np.zeros((bsize, 4))
        batch_rilts = np.zeros((bsize, 3, 3))
        batch_ks = np.zeros((bsize, 3, 3))
        batch_rot_angle = np.zeros((bsize,))
        batch_img_dims = np.zeros((bsize, 2))
        batch_one_hot_vec = np.zeros((bsize, NUM_CLASS))
        batch_new_center = np.zeros((bsize, 3))
        batch_new_heading_class = np.zeros((bsize,), dtype=np.int32)
        batch_new_heading_residual = np.zeros((bsize,))
        batch_new_size_class = np.zeros((bsize,), dtype=np.int32)
        batch_new_size_residual = np.zeros((bsize, 3))
        batch_box_iou = np.zeros((bsize,))
        batch_y_center_delta = np.zeros((bsize, 3))
        batch_y_size_delta = np.zeros((bsize, 3))
        batch_y_angle_delta = np.zeros((bsize,))

        for i in range(bsize):
            curr_data_pt_should_be_fit = (np.random.rand() < proportion_of_boxpc_fit)
            ps, img, seg, center, hclass, hres, sclass, sres, box2d, rtilt, k, rotangle, \
            img_dims, onehotvec, new_center, new_hclass, new_hres, new_sclass, new_sres, \
            box_iou, y_center_delta, y_size_delta, y_angle_delta = \
                self.get(idxs[i + start_idx], curr_data_pt_should_be_fit, 
                         boxpc_nofit_bounds=boxpc_nofit_bounds, 
                         boxpc_fit_bounds=boxpc_fit_bounds)
            batch_one_hot_vec[i] = onehotvec
            
            batch_data[i,...] = ps[:,0:num_channel]
            #batch_image[i] = img
            batch_label[i,:] = seg
            batch_center[i,:] = center
            batch_heading_class[i] = hclass
            batch_heading_residual[i] = hres
            batch_size_class[i] = sclass
            batch_size_residual[i] = sres
            batch_box2d[i] = box2d
            batch_rilts[i] = rtilt
            batch_ks[i] = k
            batch_rot_angle[i] = rotangle
            batch_img_dims[i] = img_dims
            batch_new_center[i,:] = new_center
            batch_new_heading_class[i] = new_hclass
            batch_new_heading_residual[i] = new_hres
            batch_new_size_class[i] = new_sclass
            batch_new_size_residual[i] = new_sres
            batch_box_iou[i] = box_iou
            batch_y_center_delta[i] = y_center_delta
            batch_y_size_delta[i] = y_size_delta
            batch_y_angle_delta[i] = y_angle_delta

        # Single image since images have different sizes
        # batch_image = np.expand_dims(np.array(img), axis=0)
        batch_image = None
        return batch_data, batch_image, batch_label, batch_center, batch_heading_class, \
            batch_heading_residual, batch_size_class, batch_size_residual, batch_box2d, \
            batch_rilts, batch_ks, batch_rot_angle, batch_img_dims, batch_one_hot_vec, \
            batch_new_center, batch_new_heading_class, batch_new_heading_residual, \
            batch_new_size_class, batch_new_size_residual, batch_box_iou, batch_y_center_delta, \
            batch_y_size_delta, batch_y_angle_delta

    def sample_batch(self, bsize, num_point, num_channel, proportion_of_boxpc_fit, boxpc_nofit_bounds, boxpc_fit_bounds, equal_samples_per_class=False):
        batch_data = np.zeros((bsize, num_point, num_channel))
        #batch_image = np.zeros((bsize, IMG_H, IMG_W, IMG_CHANNELS))
        batch_label = np.zeros((bsize, num_point), dtype=np.int32)
        batch_center = np.zeros((bsize, 3))
        batch_heading_class = np.zeros((bsize,), dtype=np.int32)
        batch_heading_residual = np.zeros((bsize,))
        batch_size_class = np.zeros((bsize,), dtype=np.int32)
        batch_size_residual = np.zeros((bsize, 3))
        batch_box2d = np.zeros((bsize, 4))
        batch_rilts = np.zeros((bsize, 3, 3))
        batch_ks = np.zeros((bsize, 3, 3))
        batch_rot_angle = np.zeros((bsize,))
        batch_img_dims = np.zeros((bsize, 2))
        batch_one_hot_vec = np.zeros((bsize, NUM_CLASS))
        batch_new_center = np.zeros((bsize, 3))
        batch_new_heading_class = np.zeros((bsize,), dtype=np.int32)
        batch_new_heading_residual = np.zeros((bsize,))
        batch_new_size_class = np.zeros((bsize,), dtype=np.int32)
        batch_new_size_residual = np.zeros((bsize, 3))
        batch_box_iou = np.zeros((bsize,))
        batch_y_center_delta = np.zeros((bsize, 3))
        batch_y_size_delta = np.zeros((bsize, 3))
        batch_y_angle_delta = np.zeros((bsize,))

        if equal_samples_per_class:
            # Sample approx equal number of samples per class
            cls_to_idx_map = self.cls_to_idx_map
            choices = []
            # Make use of array_split to divide into fairly equal groups
            arr = np.array_split([1] * bsize, len(cls_to_idx_map.keys()))
            random.shuffle(arr)
            for i, group in enumerate(arr):
                cls_type = cls_to_idx_map.keys()[i]
                choice_list = np.random.choice(cls_to_idx_map[cls_type], len(group), replace=True)
                choices.extend(choice_list)
        else:
            choices = np.random.choice(len(self), bsize, replace=False)
        
        for i in range(bsize):
            idx = choices[i]
            curr_data_pt_should_be_fit = (np.random.rand() < proportion_of_boxpc_fit)
            ps, img, seg, center, hclass, hres, sclass, sres, box2d, rtilt, k, rotangle, \
            img_dims, onehotvec, new_center, new_hclass, new_hres, new_sclass, new_sres, \
            box_iou, y_center_delta, y_size_delta, y_angle_delta = \
                self.get(idx, curr_data_pt_should_be_fit, 
                         boxpc_nofit_bounds=boxpc_nofit_bounds, 
                         boxpc_fit_bounds=boxpc_fit_bounds)

            batch_data[i,...] = ps[:,0:num_channel]
            #batch_image[i] = img
            batch_label[i,:] = seg
            batch_center[i,:] = center
            batch_heading_class[i] = hclass
            batch_heading_residual[i] = hres
            batch_size_class[i] = sclass
            batch_size_residual[i] = sres
            batch_box2d[i] = box2d
            batch_rilts[i] = rtilt
            batch_ks[i] = k
            batch_rot_angle[i] = rotangle
            batch_img_dims[i] = img_dims
            batch_one_hot_vec[i] = onehotvec
            batch_new_center[i,:] = new_center
            batch_new_heading_class[i] = new_hclass
            batch_new_heading_residual[i] = new_hres
            batch_new_size_class[i] = new_sclass
            batch_new_size_residual[i] = new_sres
            batch_box_iou[i] = box_iou
            batch_y_center_delta[i] = y_center_delta
            batch_y_size_delta[i] = y_size_delta
            batch_y_angle_delta[i] = y_angle_delta

        batch_image = None
        return batch_data, batch_image, batch_label, batch_center, batch_heading_class, \
            batch_heading_residual, batch_size_class, batch_size_residual, batch_box2d, \
            batch_rilts, batch_ks, batch_rot_angle, batch_img_dims, batch_one_hot_vec, \
            batch_new_center, batch_new_heading_class, batch_new_heading_residual, \
            batch_new_size_class, batch_new_size_residual, batch_box_iou, batch_y_center_delta, \
            batch_y_size_delta, batch_y_angle_delta

if __name__=='__main__':
    TRAIN_CLS = ['bed','chair','toilet','desk','bathtub']
    DATASET = BoxPCFitDataset(classes=TRAIN_CLS, 
              center_perturbation=0.8, size_perturbation=0.2, angle_perturbation=np.pi,
              npoints=2048, rotate_to_center=True, random_flip=True, random_shift=True, 
              overwritten_data_path=os.path.join('frustums', 'train_mini_aug5x.zip.pickle'))
    DATASET.get(0, True, [0.01, 0.25], [0.85, 1.0])

