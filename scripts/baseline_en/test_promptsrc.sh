#!/bin/bash

# custom config
DATA="/home/luzhihe/clip/CoOp/data"
TRAINER=PromptSRCEn

DATASET=$1
SEED=$2
GPU=$3
SUB=$4
MDIR="pre-trained model path for PromptSRC"

CFG=En_ep5_batch128
SHOTS=16
LOADEP=5

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    --load-epoch ${LOADEP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TRAINER.ENLEARN.MODEL_DIR ${MDIR}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    --load-epoch ${LOADEP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    TRAINER.ENLEARN.MODEL_DIR ${MDIR}
fi