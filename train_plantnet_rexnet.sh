#!/bin/bash

MODEL="rexnetr_300"
NAME="xp_rexnetr300_part2"
PARENT_NAME="xp_rexnetr300"
WEIGHT_FILE="results/$PARENT_NAME/${PARENT_NAME}_weights_best_acc.tar"
BATCH_SIZE=32
EPOCHS=100

python main.py  \
--lr=0.01 --batch_size=$BATCH_SIZE --mu=0.0001 --n_epochs=$EPOCHS --epoch_decay 20 40 60 80 \
--k 1 5 --model="$MODEL" --model_file="$WEIGHT_FILE" \
--pretrained --seed=4 --image_size=256 --crop_size=244 \
--root="../plantnet_data/full_data/dataset23/" --save_name="$NAME"