#!/bin/bash

# custom config
DATA="/home/luzhihe/clip/CoOp/data"
TRAINER=ZeroshotEn

DATASET=$1
GPU=$2

SHOTS=16

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/TrainingfreeEn/tf_en.yaml \
--output-dir output/${TRAINER}/${DATASET} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES all \
DATALOADER.TEST.BATCH_SIZE 800