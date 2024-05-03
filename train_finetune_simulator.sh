
sim_data='/work/cadar/Datasets/simulation_v2/train_single_obj/'
PRETRAINED="weights"
FOLDER="finetune3"

# python train.py -sim --pretrained $FOLDER/model_ts1_160000_final.pth -log $FOLDER -s $FOLDER -m ts-fl
# python train.py -sim -sdpath $sim_data --pretrained $weights -log $FOLDER -s $FOLDER -m ts-fl 
# python train.py -sim -sdpath $sim_data -log $FOLDER -s $FOLDER -m ts1
python train.py -sim -sdpath $sim_data -log $FOLDER -s $FOLDER -m ts-fl --pretrained $PRETRAINED --finetune

