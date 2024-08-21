#!/bin/ksh 
#$ -q gpu
#$ -o finetune.out

# module load python/3.10/anaconda/2023.03
# module load cuda/11.3.1
# module load cudnn/8.2.4.15-11.4/gcc/11.2.0/module-acinbtf
# module load pytorch/2.0.0/gpu

# pip install opencv-python tqdm kornia matplotlib tensorboard scipy

# cd /work/icb/fc787762/Github/DALF_Simulator/


sim_data='/work/icb/fc787762/Datasets/train_single_object/'
PRETRAINED="weights/model_ts-fl_050000.pth"
# PRETRAINED="weights/model_ts-fl_final.pth"
FOLDER="_kinect2"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" python train.py --real_data -log $FOLDER -s $FOLDER -m ts1

