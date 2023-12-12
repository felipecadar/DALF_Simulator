
FOLDER="scratch2"

# python train.py -sim --pretrained weights/model_ts-fl_final.pth -log $FOLDER -s $FOLDER -m ts1

python train.py -sim -log $FOLDER -s $FOLDER -m ts1

# find file that ends with _final.pth at the folder $FOLDER
weights=$(find $FOLDER -name "*_final.pth")

# python train.py -sim --pretrained $FOLDER/model_ts1_160000_final.pth -log $FOLDER -s $FOLDER -m ts-fl
python train.py -sim --pretrained $weights -log $FOLDER -s $FOLDER -m ts-fl