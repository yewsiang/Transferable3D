
''' Utils for training.

Author: Charles R. Qi
Date: October 2017
'''

import numpy as np
from roi_seg_box3d_dataset import NUM_CLASS, IMG_H, IMG_W, IMG_CHANNELS

##########################################################################################################
# Prepare batch for segmentation & box estimation
########################################################################################################## 

def get_batch(dataset, idxs, start_idx, end_idx, num_point, num_channel, from_rgb_detection=False):
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx, num_point, num_channel)

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

    if dataset.one_hot: batch_one_hot_vec = np.zeros((bsize,NUM_CLASS)) # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps,img,seg,center,hclass,hres,sclass,sres,box2d,rtilt,k,rotangle,img_dims,onehotvec = \
                dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,img,seg,center,hclass,hres,sclass,sres,box2d,rtilt,k,rotangle,img_dims = \
                dataset[idxs[i+start_idx]]
        
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

    # Single image since images have different sizes
    # batch_image = np.expand_dims(np.array(img), axis=0)
    if dataset.one_hot:
        return batch_data, batch_image, batch_label, batch_center, batch_heading_class, \
            batch_heading_residual, batch_size_class, batch_size_residual, batch_box2d, \
            batch_rilts, batch_ks, batch_rot_angle, batch_img_dims, batch_one_hot_vec
    else:
        return batch_data, batch_image, batch_label, batch_center, batch_heading_class, \
            batch_heading_residual, batch_size_class, batch_size_residual, batch_box2d, \
            batch_rilts, batch_ks, batch_rot_angle, batch_img_dims
    

def get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx, num_point, num_channel):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_image = np.zeros((bsize, IMG_H, IMG_W, IMG_CHANNELS))
    batch_rot_angle = np.zeros((bsize,))
    batch_prob = np.zeros((bsize,))
    if dataset.one_hot: batch_one_hot_vec = np.zeros((bsize,NUM_CLASS)) # for car,ped,cyc

    for i in range(bsize):
        if dataset.one_hot:
            ps,img,rotangle,prob,onehotvec = dataset[idxs[i+start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps,img,rotangle,prob = dataset[idxs[i+start_idx]]

        batch_data[i,...] = ps[:,0:num_channel]
        batch_image[i] = img
        batch_rot_angle[i] = rotangle
        batch_prob[i] = prob

    # Single image since images have different sizes
    # batch_image = dataset[idxs[0]]
    if dataset.one_hot:
        return batch_data, batch_image, batch_rot_angle, batch_prob, batch_one_hot_vec
    else:
        return batch_data, batch_image, batch_rot_angle, batch_prob
    