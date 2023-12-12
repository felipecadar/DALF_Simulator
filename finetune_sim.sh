
FOLDER="finetune2"

python train.py -sim --pretrained weights/model_ts-fl_final.pth -log $FOLDER -s $FOLDER -m ts-fl --finetune