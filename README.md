## Transferable Semi-supervised 3D Object Detection from RGB-D Data

![network](https://github.com/yewsiang/Transferable3D/blob/master/docs/network.jpg)

## Introduction

<center><img src="https://github.com/yewsiang/Transferable3D/blob/master/docs/teaser.jpg" width="540px"/></center>

We investigate the direction of training a 3D object detector for new object classes from only 2D bounding box labels of these new classes, while simultaneously transferring information from 3D bounding box labels of the existing classes. To this end, we propose a transferable semi-supervised 3D object detection model that learns a 3D object detector network from training data with two disjoint sets of object classes - a set of strong classes with both 2D and 3D box labels, and another set of weak classes with only 2D box labels. In particular, we suggest a relaxed reprojection loss, box prior loss and a Box-to-Point Cloud Fit network that allow us to effectively transfer useful 3D information from the strong classes to the weak classes during training, and consequently, enable the network to detect 3D objects in the weak classes during inference. Experimental results show that our proposed algorithm outperforms baseline approaches and achieves promising results compared to fully-supervised approaches on the SUN-RGBD and KITTI datasets. Furthermore, we show that our Box-to-Point Cloud Fit network improves performances of the fully-supervised approaches on both datasets.

More concretely, given two disjoint sets of classes (strong classes A and weak classes B), the objective of this work is to learn 3D object detectors to perform inference on classes B despite only having 2D + 3D bounding box labels of classes A and 2D bounding box labels of classes B, i.e. we do not require 3D bounding box labels of classes B.

## Citation

        @article{tang2019transferable3D,
          title={Transferable Semi-supervised 3D Object Detection from RGB-D Data},
          author={Tang, Yew Siang and Lee, Gim Hee},
          journal={arXiv preprint arXiv:1904.10300},
          year={2019}
        }

## Usage

### Prepare Training Data

#### SUN-RGBD

Download <a href="http://rgbd.cs.princeton.edu">SUNRGBD V1 dataset</a> and toolkit. Keep only the `Metadata` and place it in `Transferable3D/sunrgbd/SUNRGBDtoolbox/`

Open MATLAB and `cd Transferable3D/sunrgbd/SUNRGBDtoolbox/`, then run `extract_rgbd_data.m`. 
This creates the folders image, depth, calib, and label_dimension which holds the SUN RGB-D Images, Point Cloud, Calibration, Labels data respectively. 
The generated data will be stored in `Transferable3D/datasets/mysunrgbd`. You can move the `mysunrgbd` folder anywhere (you just have to provide a path to it).

Change directory with `cd Transferable3D/sunrgbd`, then run `python sunrgbd_data/sunrgbd_data.py`.
This for TensorFlow training pipeline. It will prepare frustum point clouds and labels and save them to zipped pickle files in `sunrgbd/frustums`.

### Training Semi-supervised Frustum PointNets

#### SUN-RGBD

We let Classes A = {bed, chair, toilet, desk, bathtub} and Classes B = {table, sofa, dresser, night_stand, bookshelf}. 
<b>To test on A in the semi-supervised setting, we train on 2D labels from A and 2D + 3D labels from B.</b>

Before running the code below, replace NAME_A1, NAME_B1 and NAME_C1 to your desired names. Also, in most of the code, there would be a path `/home/yewsiang/Transferable3D/` that you have to change to your own directory.

You can either run the script

```
scripts/train_semisup_bed.sh 
```

(run `chmod +x scripts/train_semisup_bed.sh` and cd to `Transferable3D/` first) or run the following 3 commands yourself:

a) Pretrain Backbone network

```
python sunrgbd_detection/train_semisup.py --model semisup_v1_sunrgbd \
--batch_size 32 --max_epoch 31 --SEMI_MODEL A \
--SUNRGBD_SEMI_TRAIN_CLS bed chair toilet desk bathtub \
--SUNRGBD_SEMI_TEST_CLS table sofa dresser night_stand bookshelf \
--log_dir [NAME_A1] --train_data trainval_aug5x \
--SEMI_SAMPLING_METHOD BATCH --SEMI_USE_LABELS2D_OF_CLASSES3D 1 \
--SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB 0 --WEAK_REPROJECTION_CLIP_PRED_BOX 0 \
--WEAK_REPROJECTION_CLIP_LOWERB_LOSS 0 --WEAK_WEIGHT_SURFACE 0 \
--WEAK_WEIGHT_REPROJECTION 0
```

b) Pretrain Box-PC Fit model

```
python sunrgbd_detection/train_boxpc.py --BOX_PC_MASK_REPRESENTATION A \
--SUNRGBD_SEMI_TRAIN_CLS bed chair toilet desk bathtub \
--SUNRGBD_SEMI_TEST_CLS table sofa dresser night_stand bookshelf \
--log_dir [NAME_B1] --train_data trainval_aug5x \ 
--BOXPC_SAMPLING_METHOD SAMPLE --BOXPC_DELTA_LOSS_TYPE huber \
--BOXPC_NOFIT_BOUNDS 0.01 0.25 --BOXPC_FIT_BOUNDS 0.7 1.0 \
--BOXPC_CENTER_PERTURBATION 0.8 --BOXPC_SIZE_PERTURBATION 0.2 \
--BOXPC_ANGLE_PERTURBATION 3.1415 --BOXPC_WEIGHT_DELTA 4
```

c) Training of Semi-supervised Model

```
python sunrgbd_detection/train_semisup_adv.py --SEMI_MODEL F \
--BOX_PC_MASK_REPRESENTATION A --max_epoch 31 --use_one_hot \
--SUNRGBD_SEMI_TRAIN_CLS bed chair toilet desk bathtub \
--SUNRGBD_SEMI_TEST_CLS table sofa dresser night_stand bookshelf \
--log_dir [NAME_C1] --train_data trainval_aug5x \
--SEMI_SAMPLING_METHOD ALTERNATE_BATCH --SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB 1 \
--init_class_ag_path experiments/[NAME_A1]/model_epoch_30.ckpt \
--init_boxpc_path experiments_boxpc/[NAME_B1]/model_epoch_30.ckpt \
--SEMI_USE_LABELS2D_OF_CLASSES3D 0 --SEMI_TRAIN_BOXPC_MODEL 0 \
--SEMI_TRAIN_BOX_TRAIN_CLASS_AG_TNET 1 --SEMI_TRAIN_BOX_TRAIN_CLASS_AG_BOX 1 \
--SEMI_BOXPC_FIT_ONLY_ON_2D_CLS 1 --SEMI_INTRACLSDIMS_ONLY_ON_2D_CLS 1 \
--WEAK_REPROJECTION_ONLY_ON_2D_CLS 1 --SEMI_WEIGHT_BOXPC_FIT_LOSS 1 \
--WEAK_WEIGHT_INTRACLASSVAR 2 --WEAK_WEIGHT_REPROJECTION 0 \
--SEMI_MULTIPLIER_FOR_WEAK_LOSS 0.05 --SEMI_BOXPC_MIN_FIT_LOSS_AFT_REFINE 1
```

<b>Similarly, we need to perform the reverse direction: To test on B in the semi-supervised setting, we train on 2D labels from B and 2D + 3D labels from A.</b> Main difference with the top is that we specify `--SUNRGBD_SEMI_TRAIN_CLS table sofa dresser night_stand bookshelf --SUNRGBD_SEMI_TEST_CLS bed chair toilet desk bathtub`.

You can either run the script

```
scripts/train_semisup_table.sh
```

 (run `chmod +x scripts/train_semisup_table.sh` and cd to `Transferable3D/` first) or run the following 3 commands yourself:

a) Pretrain Backbone network

```
python sunrgbd_detection/train_semisup.py --model semisup_v1_sunrgbd \
--batch_size 32 --max_epoch 31 --SEMI_MODEL A \
--SUNRGBD_SEMI_TRAIN_CLS table sofa dresser night_stand bookshelf \
--SUNRGBD_SEMI_TEST_CLS bed chair toilet desk bathtub \
--log_dir [NAME_A2] --train_data trainval_aug5x \
--SEMI_SAMPLING_METHOD BATCH --SEMI_USE_LABELS2D_OF_CLASSES3D 1 \
--SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB 0 --WEAK_REPROJECTION_CLIP_PRED_BOX 0 \
--WEAK_REPROJECTION_CLIP_LOWERB_LOSS 0 --WEAK_WEIGHT_SURFACE 0 \
--WEAK_WEIGHT_REPROJECTION 0
```

b) Pretrain Box-PC Fit model

```
python sunrgbd_detection/train_boxpc.py --BOX_PC_MASK_REPRESENTATION A \
--SUNRGBD_SEMI_TRAIN_CLS table sofa dresser night_stand bookshelf \
--SUNRGBD_SEMI_TEST_CLS bed chair toilet desk bathtub \
--log_dir [NAME_B2] --train_data trainval_aug5x \
--BOXPC_SAMPLING_METHOD SAMPLE --BOXPC_DELTA_LOSS_TYPE huber \
--BOXPC_NOFIT_BOUNDS 0.01 0.25 --BOXPC_FIT_BOUNDS 0.7 1.0 \
--BOXPC_CENTER_PERTURBATION 0.8 --BOXPC_SIZE_PERTURBATION 0.2 \
--BOXPC_ANGLE_PERTURBATION 3.1415 --BOXPC_WEIGHT_DELTA 4
```

c) Training of Semi-supervised Model

```
python sunrgbd_detection/train_semisup_adv.py --SEMI_MODEL F \
--BOX_PC_MASK_REPRESENTATION A --max_epoch 31 --use_one_hot \
--SUNRGBD_SEMI_TRAIN_CLS table sofa dresser night_stand bookshelf \
--SUNRGBD_SEMI_TEST_CLS bed chair toilet desk bathtub \
--log_dir [NAME_C2] --train_data trainval_aug5x \
--SEMI_SAMPLING_METHOD ALTERNATE_BATCH --SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB 0.5 \
--init_class_ag_path experiments/[NAME_A2]/model_epoch_30.ckpt \
--init_boxpc_path experiments_boxpc/[NAME_B2]/model_epoch_30.ckpt \
--SEMI_USE_LABELS2D_OF_CLASSES3D 0 --SEMI_TRAIN_BOXPC_MODEL 0 \
--SEMI_TRAIN_BOX_TRAIN_CLASS_AG_TNET 1 --SEMI_TRAIN_BOX_TRAIN_CLASS_AG_BOX 1 \
--SEMI_BOXPC_FIT_ONLY_ON_2D_CLS 1 --SEMI_INTRACLSDIMS_ONLY_ON_2D_CLS 1 \
--WEAK_REPROJECTION_ONLY_ON_2D_CLS 1 --SEMI_WEIGHT_BOXPC_FIT_LOSS 1 \
--WEAK_WEIGHT_INTRACLASSVAR 2 --WEAK_WEIGHT_REPROJECTION 0 \
--SEMI_MULTIPLIER_FOR_WEAK_LOSS 0.05 --SEMI_BOXPC_MIN_FIT_LOSS_AFT_REFINE 1
```

### Evaluation

#### SUN-RGBD

![qual_results](https://github.com/yewsiang/Transferable3D/blob/master/docs/qual_results.jpg)

To run on 2D detections, you will have to first extract the proposals using `sunrgbd/sunrgbd_data.py`, which will store it into a `zip.pickle` file. Suppose that file is `test_FasterRCNN_TrainedOnTrainval.zip.pickle`, then we:

a) Make predictions and store into [PRED_A1]

```
python sunrgbd_detection/test_semisup.py --test B \
--model semisup_v1_sunrgbd --semi_type F --use_one_hot \
--model_path experiments_adv/[NAME_C1]/model_epoch_30.ckpt \
--data_path frustums/test.zip.pickle \
--output [PRED_A1] --pred_prefix 'F2_' --refine 1 --from_rgb_detection
```

If you do not want to evaluate from RGB detections but from 2D ground truths:

```
python sunrgbd_detection/test_semisup.py --test B \
--model semisup_v1_sunrgbd --semi_type F --use_one_hot \
--model_path experiments_adv/[NAME_C1]/model_epoch_30.ckpt \
--data_path frustums/test_FasterRCNN_TrainedOnTrainval.zip.pickle \
--output [PRED_A1] --pred_prefix 'F2_' --refine 1
```

b) Evaluate the [PRED_A1] predictions using MATLAB script

Open `Transferable3D/evaluation/sunrgbd/detection/script_3Deval.m` in MATLAB, change the necessary files and variables before executing the code.

### Visualization

![vis](https://github.com/yewsiang/Transferable3D/blob/master/docs/vis.jpg)

#### SUN-RGBD

There are somne visualizations that can be ran with `sunrgbd_detection/viewer.py`. If you wish to use them, you will need to install vtk.

Possible actions while viewing the point clouds:
- Pressing `s`: takes a screenshot of the point cloud
- Pressing `g`: toggles ground truth 3D bbox
- Pressing `p`: toggles model predictions
- Pressing `c`: toggles the color of points within the box
- Pressing `i`: toggles image
- Pressing `t`: toggles text

Visualizations:

a) `python sunrgbd_detection/viewer.py --vis pc --filenum 5060`
- Shows the point cloud + image of the scene of file number 5060

b) `python sunrgbd_detection/viewer.py --vis fpc --filenum 5060`
- Shows the frustum point clouds + image of the scene of file number 5060

c) `python sunrgbd_detection/viewer.py --vis seg_box --filename frustums/train_mini.zip.pickle --num 20`
- Shows 20 randomly sampled frustums of the ROISegBoxDataset using `train_mini.zip.pickle`

d) `python sunrgbd_detection/viewer.py --vis box_pc --filename frustums/train_mini.zip.pickle --num 20`
- Shows 20 randomly sampled frustums of the BoxPCFitDataset using `train_mini.zip.pickle`

e) `python sunrgbd_detection/viewer.py --vis pred2d --pred_files predictions2D/[PRED_FILES]`
- Shows the 2D predictions of your 2D detector

f) `python sunrgbd_detection/viewer.py --vis pred3d --pred_files predictions3D/[PRED_A1].pickle --gt_file frustums/[GROUND_TRUTH].zip.pickle`
- Shows the 3D predictions of your semi-supervised 3D object detection model and performance statistics of those predictions
- Note: You have to change the `WHITE_LIST` variable at the top to the classes that you are testing on in `test_semisup.py`, or you may encounter keys errors for the untested classes.


## License
Our code is released under the Apache 2.0 license (see LICENSE file for details).

## References
* <a href="http://stanford.edu/~rqi/pointnet" target="_blank">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a> by Qi et al. (CVPR 2017 Oral Presentation). Code and data: <a href="https://github.com/charlesq34/pointnet">here</a>.
* <a href="http://stanford.edu/~rqi/pointnet2" target="_black">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017). Code and data: <a href="https://github.com/charlesq34/pointnet2">here</a>.
* <a href="http://stanford.edu/~rqi/frustum-pointnets" target="_black">Frustum PointNets for 3D Object Detection from RGB-D Data</a> by Qi et al. (CVPR 2018). Code and data: <a href="https://github.com/charlesq34/frustum-pointnets">here</a>.
