
FOLDER="scratch2"

# python train.py -sim --pretrained weights/model_ts-fl_final.pth -log $FOLDER -s $FOLDER -m ts1

# python train.py -sim -log $FOLDER -s $FOLDER -m ts1

# find file that ends with _final.pth at the folder $FOLDER
# weights=$(find $FOLDER -name "*_final.pth")
weights='/home/cadar/Documents/Github/DALF_Simulator/scratch2/model_ts1_120000.pth'
sim_data='/work/cadar/Datasets/simulation_v2/train_single_obj/'

# python train.py -sim --pretrained $FOLDER/model_ts1_160000_final.pth -log $FOLDER -s $FOLDER -m ts-fl
# python train.py -sim -sdpath $sim_data --pretrained $weights -log $FOLDER -s $FOLDER -m ts-fl 
python train.py -sim -sdpath $sim_data -log $FOLDER -s $FOLDER -m ts1