#!/bin/bash

# custom config
DATA="/home/luzhihe/clip/CoOp/data"
TRAINER=TrainingfreeEn

DATASET=$1
SEED=$2
GPU=$3
SUB=$4
WRN50=$5
WRN101=$6
WVIT32=$7

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
DATASET.SUBSAMPLE_CLASSES ${SUB} \
TRAINER.ENLEARN.SEARCH_EVAL True \
DATALOADER.TEST.BATCH_SIZE 800 \
TRAINER.ENLEARN.WEIGHT_RN50 ${WRN50} \
TRAINER.ENLEARN.WEIGHT_RN101 ${WRN101} \
TRAINER.ENLEARN.WEIGHT_VIT32 ${WVIT32}