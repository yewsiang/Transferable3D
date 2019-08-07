
""" Dataset for training of semi-supervised 3D models.

Author: Tang Yew Siang
Date: July 2019
"""

import os
import sys
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../sunrgbd_data'))
sys.path.append(os.path.join(BASE_DIR, '../../train'))

import numpy as np
from box_util import box3d_iou
from utils import roty, load_zipped_pickle

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
IMG_H, IMG_W = 300, 300
IMG_CHANNELS = 3

def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from  
        class center angle to current angle.
       
        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    angle = angle%(2*np.pi)
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = int(shifted_angle/angle_per_class)
    residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
    return class_id, residual_angle

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi:
        angle = angle - 2*np.pi
    return angle
        
def size2class(size, type_name):
    ''' Convert 3D box size (l,w,h) to size class and size residual '''
    size_class = type2class[type_name]
    size_residual = size - type_mean_size[type_name]
    return size_class, size_residual

def class2size(pred_cls, residual):
    ''' Inverse function to size2class '''
    mean_size = type_mean_size[class2type[pred_cls]]
    return mean_size + residual

def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def compute_box3d_iou(center_pred, heading_logits, heading_residuals, size_logits, size_residuals, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label):
    ''' Used for confidence score supervision..
    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1) # B
    heading_residual = np.array([heading_residuals[i,heading_class[i]] for i in range(batch_size)]) # B,
    size_class = np.argmax(size_logits, 1) # B
    size_residual = np.vstack([size_residuals[i,size_class[i],:] for i in range(batch_size)])

    iou2d_list = [] 
    iou3d_list = [] 
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i], heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i], heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label, heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label) 
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), np.array(iou3d_list, dtype=np.float32)

def compare_with_anchor_boxes(center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label):
    ''' Compute IoUs between GT box and anchor boxes.
        Compute heading,size,center regression from anchor boxes to GT box: NHxNS of them in the order of
            heading0: size0,size1,...
            heading1: size0,size1,...
            ...
    Inputs:
        center_label: (B,3) -- assume this center is already close to (0,0,0) e.g. subtracted stage1_center
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,K) where K = NH*NS
        iou3ds: (B,K) 
        center_residuals: (B,K,3)
        heading_residuals: (B,K)
        size_residuals: (B,K,3)
    '''
    B = len(heading_class_label)
    K = NUM_HEADING_BIN*NUM_SIZE_CLUSTER
    iou3ds = np.zeros((B,K), dtype=np.float32)
    iou2ds = np.zeros((B,K), dtype=np.float32)
    center_residuals = np.zeros((B,K,3), dtype=np.float32)
    heading_residuals = np.zeros((B,K), dtype=np.float32)
    size_residuals = np.zeros((B,K,3), dtype=np.float32)
 
    corners_3d_anchor_list = []
    heading_anchor_list = []
    box_anchor_list = []
    for j in range(NUM_HEADING_BIN):
       for k in range(NUM_SIZE_CLUSTER):
           heading_angle = class2angle(j,0,NUM_HEADING_BIN)
           box_size = class2size(k,np.zeros((3,)))
           corners_3d_anchor = get_3d_box(box_size, heading_angle, np.zeros((3,)))
           corners_3d_anchor_list.append(corners_3d_anchor)
           heading_anchor_list.append(heading_angle)
           box_anchor_list.append(box_size)

    for i in range(B):
        heading_angle_label = class2angle(heading_class_label[i], heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label, heading_angle_label, center_label[i])
        for j in range(K):
            iou_3d, iou_2d = box3d_iou(corners_3d_anchor_list[j], corners_3d_label) 
            iou3ds[i,j] = iou_3d
            iou2ds[i,j] = iou_2d
            center_residuals[i,j,:] = center_label[i]
            heading_residuals[i,j] = heading_angle_label - heading_anchor_list[j]
            size_residuals[i,j,:] = box_size_label - box_anchor_list[j]

    return iou2ds, iou3ds, center_residuals, heading_residuals, size_residuals


class ROISemiDataset(object):
    def __init__(self, classes3D, classes2D, npoints, data3D_keep_prob=1, add3D_for_classes2D_prob=0,
        random_flip=False, random_shift=False, rotate_to_center=False, overwritten_data_path=None):
        # Classes3D are the classes that we want the 3D bounding boxes of while
        # Classes2D are the classes that we want the 2D bounding boxes of.
        self.classes3D = classes3D
        self.classes2D = classes2D
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        assert(overwritten_data_path is not None)

        idx_l, box2d_l, box3d_l, image_crop_l, points_l, label_l, cls_type_l, heading_l, size_l, \
            rtilt_l, k_l, frustum_angle_l, img_dims_l = load_zipped_pickle(overwritten_data_path)

        # Maps cls_type to it's idx within (self.label_3Dl, self.heading_3Dl, etc) to allow class-wise sampling
        self.cls_to_idx_map3D = {} 
        # Choosing the Classes to provide 3D bounding boxes for
        self.idx_3Dl, self.box2d_3Dl, self.box3d_3Dl, self.image_crop_3Dl, self.points_3Dl, self.label_3Dl, \
            self.cls_type_3Dl, self.heading_3Dl, self.size_3Dl, self.rtilt_3Dl, self.k_3Dl, \
            self.frustum_angle_3Dl, self.img_dims_3Dl = [], [], [], [], [], [], [], [], [], [], [], [], []

        # Maps cls_type to it's idx within (self.label_2Dl, self.heading_2Dl, etc) to allow class-wise sampling
        self.cls_to_idx_map2D = {}
        # Choosing the Classes to only provide 2D bounding boxes for
        self.idx_2Dl, self.box2d_2Dl, self.box3d_2Dl, self.image_crop_2Dl, self.points_2Dl, self.label_2Dl, \
            self.cls_type_2Dl, self.heading_2Dl, self.size_2Dl, self.rtilt_2Dl, self.k_2Dl, \
            self.frustum_angle_2Dl, self.img_dims_2Dl = [], [], [], [], [], [], [], [], [], [], [], [], []

        np.random.seed(20)
        cls3D_idx, cls2D_idx = 0, 0
        for idx, box2d, box3d, image_crop, points, label, cls_type, heading, size, rtilt, k, \
            frustum_angle, img_dims in zip(idx_l, box2d_l, box3d_l, image_crop_l, points_l, \
            label_l, cls_type_l, heading_l, size_l, rtilt_l, k_l, frustum_angle_l, img_dims_l):
            
            if (cls_type in self.classes3D and (np.random.rand() <= data3D_keep_prob)) or \
               (np.random.rand() < add3D_for_classes2D_prob):

                if self.cls_to_idx_map3D.get(cls_type) is None:
                    self.cls_to_idx_map3D[cls_type] = [cls3D_idx]
                else:
                    self.cls_to_idx_map3D[cls_type].append(cls3D_idx)
                cls3D_idx += 1

                self.idx_3Dl.append(idx)
                self.box2d_3Dl.append(box2d)
                self.box3d_3Dl.append(box3d)
                #self.image_crop_3Dl.append(image_crop)
                self.points_3Dl.append(points)
                self.label_3Dl.append(label)
                self.cls_type_3Dl.append(cls_type)
                self.heading_3Dl.append(heading)
                self.size_3Dl.append(size)
                self.rtilt_3Dl.append(rtilt)
                self.k_3Dl.append(k)
                self.frustum_angle_3Dl.append(frustum_angle)
                self.img_dims_3Dl.append(img_dims)

            if self.classes2D is not None and cls_type in self.classes2D:
                if self.cls_to_idx_map2D.get(cls_type) is None:
                    self.cls_to_idx_map2D[cls_type] = [cls2D_idx]
                else:
                    self.cls_to_idx_map2D[cls_type].append(cls2D_idx)
                cls2D_idx += 1

                self.idx_2Dl.append(idx)
                self.box2d_2Dl.append(box2d)
                self.box3d_2Dl.append(box3d)
                #self.image_crop_2Dl.append(image_crop)
                self.points_2Dl.append(points)
                self.label_2Dl.append(label)
                self.cls_type_2Dl.append(cls_type)
                self.heading_2Dl.append(heading)
                self.size_2Dl.append(size)
                self.rtilt_2Dl.append(rtilt)
                self.k_2Dl.append(k)
                self.frustum_angle_2Dl.append(frustum_angle)
                self.img_dims_2Dl.append(img_dims)

    def __len__(self):
        return self.get_len_classes3D() + self.get_len_classes2D()

    # ============================== 3D ==============================
    # These are the classes for which we have 2D + 3D bounding box labels
    def get_len_classes3D(self):
        return len(self.points_3Dl)

    def get_classes3D(self, index):
        # ------------------------------ INPUTS ----------------------------
        box2d = self.box2d_3Dl[index]
        rtilt = self.rtilt_3Dl[index]
        k = self.k_3Dl[index]
        rot_angle = self.get_center_view_rot_angle_3D(index)
        img_dims = self.img_dims_3Dl[index]

        # compute one hot vector
        cls_type = self.cls_type_3Dl[index]
        #assert(cls_type in self.classes3D)
        one_hot_vec = np.zeros((NUM_CLASS))
        one_hot_vec[type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set_3D(index)
        else:
            point_set = self.points_3Dl[index]
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # Get image crop
        image = np.zeros((IMG_W, IMG_H, 3))
        # image = self.image_crop_3Dl[index] 
        # image = cv2.resize(image, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR) 

        
        # ------------------------------ LABELS ----------------------------
        seg = self.label_3Dl[index] 
        seg = seg[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center_3D(index)
        else:
            box3d_center = self.get_box3d_center_3D(index)

        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_3Dl[index] - rot_angle
        else:
            heading_angle = self.heading_3Dl[index]

        # Size
        size_class, size_residual = size2class(self.size_3Dl[index], self.cls_type_3Dl[index])

        # Data Augmentation
        if self.random_flip:
            if np.random.random()>0.5:
                point_set[:,0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
                # NOTE: rot_angle won't be correct if we have random_flip...
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
            shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
            point_set[:,2] += shift
            box3d_center[2] += shift
            height_shift = np.random.random()*0.4-0.2 # randomly shift +-0.2 meters
            point_set[:,1] += height_shift
            box3d_center[1] += height_shift

        angle_class, angle_residual = angle2class(heading_angle, NUM_HEADING_BIN)

        return point_set, image, seg, box3d_center, angle_class, angle_residual, size_class, \
            size_residual, box2d, rtilt, k, rot_angle, img_dims, one_hot_vec

    def get_center_view_rot_angle_3D(self, index):
        return np.pi/2.0 + self.frustum_angle_3Dl[index]

    def get_box3d_center_3D(self, index):
        box3d_center = (self.box3d_3Dl[index][0,:] + self.box3d_3Dl[index][6,:])/2.0
        return box3d_center

    def get_center_view_box3d_center_3D(self, index):
        box3d_center = (self.box3d_3Dl[index][0,:] + self.box3d_3Dl[index][6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), self.get_center_view_rot_angle_3D(index)).squeeze()
        
    def get_center_view_box3d_3D(self, index):
        box3d = self.box3d_3Dl[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, self.get_center_view_rot_angle_3D(index))

    def get_center_view_point_set_3D(self, index):
        ''' Input ps is NxC points with first 3 channels as XYZ
            z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.points_3Dl[index])
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle_3D(index))


    # ============================== 2D ==============================
    # These are the classes for which we only have the 2D bounding box labels
    def get_len_classes2D(self):
        return len(self.points_2Dl)

    def get_classes2D(self, index):
        # ------------------------------ INPUTS ----------------------------
        box2d = self.box2d_2Dl[index]
        rtilt = self.rtilt_2Dl[index]
        k = self.k_2Dl[index]
        rot_angle = self.get_center_view_rot_angle_2D(index)
        img_dims = self.img_dims_2Dl[index]

        # compute one hot vector
        cls_type = self.cls_type_2Dl[index]
        assert(cls_type in self.classes2D)
        one_hot_vec = np.zeros((NUM_CLASS))
        one_hot_vec[type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set_2D(index)
        else:
            point_set = self.points_2Dl[index]
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # Get image crop
        image = np.zeros((IMG_W, IMG_H, 3))
        # image = self.image_crop_2Dl[index] 
        # image = cv2.resize(image, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR) 

        
        # ------------------------------ LABELS ----------------------------
        seg = self.label_2Dl[index] 
        seg = seg[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center_2D(index)
        else:
            box3d_center = self.get_box3d_center_2D(index)

        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_2Dl[index] - rot_angle
        else:
            heading_angle = self.heading_2Dl[index]

        # Size
        size_class, size_residual = size2class(self.size_2Dl[index], self.cls_type_2Dl[index])

        # 2D Classes cannot be augmented because the projection will no longer be accurate
        # # Data Augmentation
        # if self.random_flip:
        #     if np.random.random()>0.5:
        #         point_set[:,0] *= -1
        #         box3d_center[0] *= -1
        #         heading_angle = np.pi - heading_angle
        #         # NOTE: rot_angle won't be correct if we have random_flip...
        # if self.random_shift:
        #     dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
        #     shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
        #     point_set[:,2] += shift
        #     box3d_center[2] += shift
        #     height_shift = np.random.random()*0.4-0.2 # randomly shift +-0.2 meters
        #     point_set[:,1] += height_shift
        #     box3d_center[1] += height_shift

        angle_class, angle_residual = angle2class(heading_angle, NUM_HEADING_BIN)

        # return point_set, image, seg, box3d_center, angle_class, angle_residual, size_class, \
        #     size_residual, box2d, rtilt, k, rot_angle, img_dims, one_hot_vec
        return point_set, image, np.zeros_like(seg), np.zeros_like(box3d_center), \
            np.zeros_like(angle_class), np.zeros_like(angle_residual), np.zeros_like(size_class), \
            np.zeros_like(size_residual), box2d, rtilt, k, rot_angle, img_dims, one_hot_vec

    def get_center_view_rot_angle_2D(self, index):
        return np.pi/2.0 + self.frustum_angle_2Dl[index]

    def get_box3d_center_2D(self, index):
        box3d_center = (self.box3d_2Dl[index][0,:] + self.box3d_2Dl[index][6,:])/2.0
        return box3d_center

    def get_center_view_box3d_center_2D(self, index):
        box3d_center = (self.box3d_2Dl[index][0,:] + self.box3d_2Dl[index][6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), self.get_center_view_rot_angle_2D(index)).squeeze()
        
    def get_center_view_box3d_2D(self, index):
        box3d = self.box3d_2Dl[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, self.get_center_view_rot_angle_2D(index))

    def get_center_view_point_set_2D(self, index):
        ''' Input ps is NxC points with first 3 channels as XYZ
            z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.points_2Dl[index])
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle_2D(index))


    # ============================== Get Data ==============================
    def get_batch(self, idxs, start_idx, end_idx, num_point, num_channel):

        bsize = end_idx-start_idx
        batch_data = np.zeros((bsize, num_point, num_channel))
        batch_image = np.zeros((bsize, IMG_H, IMG_W, IMG_CHANNELS))
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
        batch_is_data_2D = np.zeros((bsize,), dtype=np.int32) # Indicate 1 if it comes from the 2D list

        len_classes3D = self.get_len_classes3D()
        for i in range(bsize):
            idx = idxs[i+start_idx]
            # Place 3D data at the front [0, len_classes3D) and 
            # 2D data at the back [len_classes3D, len_classes3D + len_classes2D)
            is_data_2D = 1 if idx >= len_classes3D else 0
            if is_data_2D:
                # Need to offset index because first len_classes3D are by labels from 3D list
                ps, img, seg, center, hclass, hres, sclass, sres, box2d, rtilt, k, rotangle, \
                    img_dims, onehotvec = self.get_classes2D(idx - len_classes3D)
            else:
                ps, img, seg, center, hclass, hres, sclass, sres, box2d, rtilt, k, rotangle, \
                    img_dims, onehotvec = self.get_classes3D(idx)

            batch_data[i,...] = ps[:,0:num_channel]
            batch_image[i] = img
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
            batch_is_data_2D[i] = is_data_2D

        return batch_data, batch_image, batch_label, batch_center, batch_heading_class, \
            batch_heading_residual, batch_size_class, batch_size_residual, batch_box2d, \
            batch_rilts, batch_ks, batch_rot_angle, batch_img_dims, batch_one_hot_vec, \
            batch_is_data_2D

    def sample_from_set(self, sample_from_2D, bsize, num_point, num_channel, equal_samples_per_class=False):
        batch_data = np.zeros((bsize, num_point, num_channel))
        batch_image = np.zeros((bsize, IMG_H, IMG_W, IMG_CHANNELS))
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
        batch_is_data_2D = np.ones((bsize,), dtype=np.int32) if sample_from_2D else \
                           np.zeros((bsize,), dtype=np.int32) # Indicate 1 if it comes from the 2D list

        sampling_fn = self.get_classes2D       if sample_from_2D else self.get_classes3D
        len_classes = self.get_len_classes2D() if sample_from_2D else self.get_len_classes3D()

        if equal_samples_per_class:
            # Sample approx equal number of samples per class
            cls_to_idx_map = self.cls_to_idx_map2D if sample_from_2D else self.cls_to_idx_map3D
            choices = []
            # Make use of array_split to divide into fairly equal groups
            arr = np.array_split([1] * bsize, len(cls_to_idx_map.keys()))
            random.shuffle(arr)
            for i, group in enumerate(arr):
                cls_type = cls_to_idx_map.keys()[i]
                choice_list = np.random.choice(cls_to_idx_map[cls_type], len(group), replace=True)
                choices.extend(choice_list)
        else:
            choices = np.random.choice(len_classes, bsize, replace=False)
        
        for i in range(bsize):
            idx = choices[i]
            ps, img, seg, center, hclass, hres, sclass, sres, box2d, rtilt, k, rotangle, \
                img_dims, onehotvec = sampling_fn(idx)

            batch_data[i,...] = ps[:,0:num_channel]
            batch_image[i] = img
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

        return batch_data, batch_image, batch_label, batch_center, batch_heading_class, \
            batch_heading_residual, batch_size_class, batch_size_residual, batch_box2d, \
            batch_rilts, batch_ks, batch_rot_angle, batch_img_dims, batch_one_hot_vec, \
            batch_is_data_2D

    def sample_pure_from_2D_cls(self, bsize, num_point, num_channel, equal_samples_per_class=False):
        # Only retrieve samples from the 2D set
        return self.sample_from_set(True, bsize, num_point, num_channel, 
                                    equal_samples_per_class=equal_samples_per_class)

    def sample_pure_from_3D_cls(self, bsize, num_point, num_channel, equal_samples_per_class=False):
        # Only retrieve samples from the 3D set
        return self.sample_from_set(False, bsize, num_point, num_channel, 
                                    equal_samples_per_class=equal_samples_per_class)

    def sample_mixed(self, bsize, num_point, num_channel, equal_samples_per_class=False):
        assert(bsize % 2 == 0)
        b2D_data, b2D_image, b2D_label, b2D_center, b2D_heading_class, b2D_heading_residual, \
        b2D_size_class, b2D_size_residual, b2D_box2d, b2D_rilts, b2D_ks, b2D_rot_angle, \
        b2D_img_dims, b2D_one_hot_vec, b2D_is_data_2D = \
            self.sample_from_set(True, bsize / 2, num_point, num_channel, 
                                 equal_samples_per_class=equal_samples_per_class)
        b3D_data, b3D_image, b3D_label, b3D_center, b3D_heading_class, b3D_heading_residual, \
        b3D_size_class, b3D_size_residual, b3D_box2d, b3D_rilts, b3D_ks, b3D_rot_angle, \
        b3D_img_dims, b3D_one_hot_vec, b3D_is_data_2D = \
            self.sample_from_set(False, bsize / 2, num_point, num_channel, 
                                 equal_samples_per_class=equal_samples_per_class)

        b_data             = np.concatenate([b2D_data, b3D_data])
        b_image            = np.concatenate([b2D_image, b3D_image])
        b_label            = np.concatenate([b2D_label, b3D_label])
        b_center           = np.concatenate([b2D_center, b3D_center])
        b_heading_class    = np.concatenate([b2D_heading_class, b3D_heading_class])
        b_heading_residual = np.concatenate([b2D_heading_residual, b3D_heading_residual])
        b_size_class       = np.concatenate([b2D_size_class, b3D_size_class])
        b_size_residual    = np.concatenate([b2D_size_residual, b3D_size_residual])
        b_box2d            = np.concatenate([b2D_box2d, b3D_box2d])
        b_rilts            = np.concatenate([b2D_rilts, b3D_rilts])
        b_ks               = np.concatenate([b2D_ks, b3D_ks])
        b_rot_angle        = np.concatenate([b2D_rot_angle, b3D_rot_angle])
        b_img_dims         = np.concatenate([b2D_img_dims, b3D_img_dims])
        b_one_hot_vec      = np.concatenate([b2D_one_hot_vec, b3D_one_hot_vec])
        b_is_data_2D       = np.concatenate([b2D_is_data_2D, b3D_is_data_2D])

        return b_data, b_image, b_label, b_center, b_heading_class, b_heading_residual, \
            b_size_class, b_size_residual, b_box2d, b_rilts, b_ks, b_rot_angle, \
            b_img_dims, b_one_hot_vec, b_is_data_2D

def from_prediction_to_label_format(center, angle_class, angle_res, size_class, size_res, rot_angle):
    l,w,h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx,ty,tz = rotate_pc_along_y(np.expand_dims(center,0),-rot_angle).squeeze()
    ty += h/2.0
    return h,w,l,tx,ty,tz,ry
