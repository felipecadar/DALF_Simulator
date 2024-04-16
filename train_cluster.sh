#!/bin/ksh 
#$ -q gpu
#$ -o result.out

module load python/3.10/anaconda/2023.03
module load cuda/11.3.1
module load cudnn/8.2.4.15-11.4/gcc/11.2.0/module-acinbtf
module load pytorch/2.0.0/gpu

pip install opencv-python tqdm kornia matplotlib tensorboard scipy

cd /work/icb/fc787762/Github/DALF_Simulator/

FOLDER="scratch2"

# python train.py -sim --pretrained weights/model_ts-fl_final.pth -log $FOLDER -s $FOLDER -m ts1
# python train.py -sim -log $FOLDER -s $FOLDER -m ts1

# find file that ends with _final.pth at the folder $FOLDER
# finf the last created file that ends with .pth
weights=$(find $FOLDER -type f -name "model_ts1_*.pth" | sort -n | tail -n 1)
echo $weights

# weights='/home/cadar/Documents/Github/DALF_Simulator/scratch2/model_ts1_120000.pth'
sim_data='/work/icb/fc787762/Datasets/train_single_object/'

# python train.py -sim --pretrained $FOLDER/model_ts1_160000_final.pth -log $FOLDER -s $FOLDER -m ts-fl
# python train.py -sim -sdpath $sim_data --pretrained $weights -log $FOLDER -s $FOLDER -m ts-fl 
# python train.py -sim -sdpath $sim_data -log $FOLDER -s $FOLDER -m ts1
python train.py -sim -sdpath $sim_data -log $FOLDER -s $FOLDER -m ts-fl --pretrained $weights

