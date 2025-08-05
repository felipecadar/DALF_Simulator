#!/bin/ksh 
#$ -q gpu
#$ -o result_finetune_new_triggers.out

module load python/3.10/anaconda/2023.03
module load cuda/11.3.1
module load cudnn/8.2.4.15-11.4/gcc/11.2.0/module-acinbtf
module load pytorch/2.0.0/gpu

pip install opencv-python tqdm kornia matplotlib tensorboard scipy

cd /work/icb/fc787762/Github/DALF_Simulator/

FOLDER="finetune_new_triggers"

# python train.py -sim --pretrained weights/model_ts-fl_final.pth -log $FOLDER -s $FOLDER -m ts1
# python train.py -sim -log $FOLDER -s $FOLDER -m ts1

# find file that ends with _final.pth at the folder $FOLDER
# finf the last created file that ends with .pth
# PRETRAINED_FOLDER="scratch2"
# weights=$(find $PRETRAINED_FOLDER -type f -name "model_ts1_*.pth" | sort -n | tail -n 1)
# weights=$(realpath $weights)
weights=./weights/model_ts-fl_final.pth
echo " >> Weights: $weights"

sim_data='/lustre/fsn1/projects/rech/xab/uus28wc/Datasets/train_single_obj'

# python train.py -sim --pretrained $FOLDER/model_ts1_160000_final.pth -log $FOLDER -s $FOLDER -m ts-fl
# python train.py -sim -sdpath $sim_data --pretrained $weights -log $FOLDER -s $FOLDER -m ts-fl 
# python train.py -sim -sdpath $sim_data -log $FOLDER -s $FOLDER -m ts1
python train.py -sim -sdpath $sim_data -log $FOLDER -s $FOLDER -m ts-fl --pretrained $weights

