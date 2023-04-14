#!/usr/bin/env bash
# e.g.: ./train_calixto_baseline.sh 0 multi30k multi30k-test2016 1234 0
#       ./train_calixto_baseline.sh 0 d d 1234 0
# 1: gpu_id
# 2: train data
# 3: test data
# 4: random_seed
# 5: suffix
PROJECT_DIR=.
export CUDA_VISIBLE_DEVICES=${1};echo "CUDA_VISIBLE_DEVICES=${1}"
TRAINDATA=${2}
# options: gmnmt, multi30k.parsed default: multi30k.parsed
if [[ "${TRAINDATA}" != "gmnmt" ]]; then
    TRAINDATA=multi30k.parsed
fi
if [[ "${TRAINDATA}" == "gmnmt" ]]; then
    TRAIN_PREFIX=gm.
else
    TRAIN_PREFIX=""
fi
TESTDATA=${3}
# options: multi30k-test2016, multi30k-test2017, mscoco-test2017
if [[ "${TESTDATA}" != "multi30k-test2017" && "${TESTDATA}" != "mscoco-test2017" ]]; then
    TESTDATA=multi30k-test2016
fi
RANDOM_SEED=${4}
if [[ "${RANDOM_SEED}" == "" ]]; then
    RANDOM_SEED=1234
fi
SUFFIX=${5}
if [[ "${SUFFIX}" == "" ]]; then
    SUFFIX=0
fi
ADDITIONAL=${6}

DATA_CONFIG=${PROJECT_DIR}/configurations/data_configurations
#DATA_CONFIG=${DATA_CONFIG}/${TRAINDATA}.${TESTDATA}.config.json
DATA_CONFIG=${DATA_CONFIG}/${TRAINDATA}.parsed.${TESTDATA}.config.json
MODEL_CONFIG=${PROJECT_DIR}/configurations/model_configurations
MODEL_CONFIG=${MODEL_CONFIG}/std-calixto-rnn-baseline-config.json

SUFFIX=${TRAIN_PREFIX}0.3drop.initdec.r${RANDOM_SEED}.${SUFFIX}
#OUT_DIR=${PROJECT_DIR}/experiment/models/nmt
OUT_DIR=${PROJECT_DIR}/experiment/models/nmt.en2cs
OUT_DIR=${OUT_DIR}/calixto_baseline.1layerlstm.${SUFFIX}
#OUT_DIR=experiment/models/nmt/calixto_baseline.1layerlstm.gm.0.3drop.initdec.r${RANDOM_SEED}.${SUFFIX}
mkdir -p ${OUT_DIR}
cp $0 ${OUT_DIR}/
nohup python ${PROJECT_DIR}/train.py \
    --config_file=${MODEL_CONFIG} \
    --dataset_config_file=${DATA_CONFIG} \
    --out_dir=${OUT_DIR} \
    --random_seed=${RANDOM_SEED} \
    --apply_torchtext=False \
    --gpu_id=${1} \
    --additional=${ADDITIONAL} \
	> ${OUT_DIR}/log.txt 2>&1 &

