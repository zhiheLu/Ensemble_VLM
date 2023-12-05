#!/bin/bash

# custom config
DATA="/home/luzhihe/clip/CoOp/data"
TRAINER=CoCoOpEn

DATASET=$1
SEED=$2
GPU=$3
MDIR="pre-trained model path for CoCoOp"

CFG=En_ep5_batch128
SHOTS=16

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.ENLEARN.MODEL_DIR ${MDIR}
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.ENLEARN.MODEL_DIR ${MDIR}
fi