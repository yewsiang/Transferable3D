
NAME_A1='TrainBed_Backbone'
NAME_B1='TrainBed_Boxpc'
NAME_C1='TrainBed_Model'

cd sunrgbd

python sunrgbd_detection/train_semisup.py --model semisup_v1_sunrgbd \
--batch_size 32 --max_epoch 31 --SEMI_MODEL A \
--SUNRGBD_SEMI_TRAIN_CLS bed chair toilet desk bathtub \
--SUNRGBD_SEMI_TEST_CLS table sofa dresser night_stand bookshelf \
--log_dir $NAME_A1 --train_data trainval_aug5x \
--SEMI_SAMPLING_METHOD BATCH --SEMI_USE_LABELS2D_OF_CLASSES3D 1 \
--SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB 0 --WEAK_REPROJECTION_CLIP_PRED_BOX 0 \
--WEAK_REPROJECTION_CLIP_LOWERB_LOSS 0 --WEAK_WEIGHT_SURFACE 0 \
--WEAK_WEIGHT_REPROJECTION 0

python sunrgbd_detection/train_boxpc.py --BOX_PC_MASK_REPRESENTATION A \
--SUNRGBD_SEMI_TRAIN_CLS bed chair toilet desk bathtub \
--SUNRGBD_SEMI_TEST_CLS table sofa dresser night_stand bookshelf \
--log_dir $NAME_B1 --train_data trainval_aug5x --max_epoch 31 \
--BOXPC_SAMPLING_METHOD SAMPLE --BOXPC_DELTA_LOSS_TYPE huber \
--BOXPC_NOFIT_BOUNDS 0.01 0.25 --BOXPC_FIT_BOUNDS 0.7 1.0 \
--BOXPC_CENTER_PERTURBATION 0.8 --BOXPC_SIZE_PERTURBATION 0.2 \
--BOXPC_ANGLE_PERTURBATION 3.1415 --BOXPC_WEIGHT_DELTA 4

python sunrgbd_detection/train_semisup_adv.py --SEMI_MODEL F \
--BOX_PC_MASK_REPRESENTATION A --max_epoch 31 --use_one_hot \
--SUNRGBD_SEMI_TRAIN_CLS bed chair toilet desk bathtub \
--SUNRGBD_SEMI_TEST_CLS table sofa dresser night_stand bookshelf \
--log_dir $NAME_C1 --train_data trainval_aug5x \
--SEMI_SAMPLING_METHOD ALTERNATE_BATCH --SEMI_SAMPLE_EQUAL_CLASS_WITH_PROB 1 \
--init_class_ag_path experiments/$NAME_A1/model_epoch_0.ckpt \
--init_boxpc_path experiments_boxpc/$NAME_B1/model_epoch_0.ckpt \
--SEMI_USE_LABELS2D_OF_CLASSES3D 0 --SEMI_TRAIN_BOXPC_MODEL 0 \
--SEMI_TRAIN_BOX_TRAIN_CLASS_AG_TNET 1 --SEMI_TRAIN_BOX_TRAIN_CLASS_AG_BOX 1 \
--SEMI_BOXPC_FIT_ONLY_ON_2D_CLS 1 --SEMI_INTRACLSDIMS_ONLY_ON_2D_CLS 1 \
--WEAK_REPROJECTION_ONLY_ON_2D_CLS 1 --SEMI_WEIGHT_BOXPC_FIT_LOSS 1 \
--WEAK_WEIGHT_INTRACLASSVAR 2 --WEAK_WEIGHT_REPROJECTION 0 \
--SEMI_MULTIPLIER_FOR_WEAK_LOSS 0.05 --SEMI_BOXPC_MIN_FIT_LOSS_AFT_REFINE 1

cd -