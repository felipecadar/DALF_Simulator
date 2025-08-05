#!/bin/bash
#SBATCH --job-name="DALF-sim"
#SBATCH --mail-user=felipecadarchamone@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="log_%j.out" # out file name
#SBATCH --error="log_%j.err" # error file name
#SBATCH --signal=USR1@60

### Use this for a 1x A100 node
##SBATCH --time=20:00:00
##SBATCH --account=xab@a100
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu_p5
##SBATCH -C a100
##SBATCH --ntasks=1 # nbr of MPI tasks (= nbr of GPU)
##SBATCH --ntasks-per-node=1 # nbr of task per node

### Use this for a 1x H100 node
#SBATCH --time=20:00:00
#SBATCH --account=xab@h100
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --ntasks=1 # nbr of MPI tasks (= nbr of GPU)
#SBATCH --ntasks-per-node=1 # nbr of task per node

# Use this for a 1x V100 32G node
##SBATCH --time=20:00:00
##SBATCH --gres=gpu:1
##SBATCH -C v100-32g
##SBATCH --account xab@v100
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1

echo '-------------------------------------'
echo "Start : $0"
echo '-------------------------------------'
echo "Job id : $SLURM_JOB_ID"
echo "Job name : $SLURM_JOB_NAME"
echo "Job node list : $SLURM_JOB_NODELIST"
echo '--------------------------------------'

module purge
# load this if running the 8x A100 node

module load arch/h100
# module load arch/a100

module load libjpeg-turbo/2.1.3
module load pytorch-gpu/py3/2.3.1
pip install --user --no-cache-dir opencv-python # opencv-contrib-python

set -x 

DATASET=$SCRATCH/Datasets/train_single_obj
python3 modules/dataset/make_masks.py --data $DATASET

DATASET=$SCRATCH/Datasets/train_multiple_obj
python3 modules/dataset/make_masks.py --data $DATASET