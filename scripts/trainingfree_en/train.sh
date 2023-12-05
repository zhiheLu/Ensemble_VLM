#!/bin/bash

# custom config
DATA="/home/luzhihe/clip/CoOp/data"
TRAINER=TrainingfreeEn

DATASET=$1
SEED=$2
GPU=$3

SHOTS=16

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/TrainingfreeEn/tf_en.yaml \
--output-dir output/${TRAINER}/${DATASET}/seed${SEED} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES base \
TRAINER.ENLEARN.WEIGHTS_SEARCH True \
DATALOADER.TRAIN_X.BATCH_SIZE 800