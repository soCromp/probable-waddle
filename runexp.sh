#!/bin/bash
name=`date +"%Y-%m-%d_%H.%M.%S"`

data=$1
corr=$2
gpu=$3

DICE=(
    0 
    # 10 25 40 50 60 75 
    90
)

echo $name
echo "Train"
python train_bg.py --gpu-ids $gpu --in-dataset $data --model resnet18 --epochs 30 --save-epoch 30 --data_label_correlation $corr --domain-num 4 --method erm --name $name  --lr 0.001 --weight-decay 0.001
echo "Get activations"
python get_activations.py --gpu-ids $gpu --in-dataset $data --model resnet18 --test_epochs 30 --data_label_correlation $corr --method erm --name $name  --root_dir datasets/ood_datasets 

for d in "${DICE[@]}"; do
    echo "Test all units, dice= $d"
    python test_bg.py --gpu-ids $gpu --in-dataset $data --model resnet18 --test_epochs 30 --data_label_correlation $corr --method erm --name $name  --root_dir datasets/ood_datasets -d $d
    echo "Present results, dice= $d"
    python present_results.py --in-dataset $data --name $name  --test_epochs 30 -d $d
done

echo $name
