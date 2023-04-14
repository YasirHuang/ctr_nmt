#!/usr/bin/env bash
# e.g.: ./infer_transformer.sh 0 multi30k multi30k-test2016 1234 0
#       ./infer_transformer.sh 0 d d 1234 0
# 1: gpu_id
# 2: train data
# 3: test data
# 4: random_seed
# 5: suffix
PROJECT_DIR=.
export CUDA_VISIBLE_DEVICES=${1};echo "CUDA_VISIBLE_DEVICES=${1}"
TRAINDATA=${2}
# options: gmnmt, multi30k. default: multi30k
if [[ "${TRAINDATA}" != "gmnmt" ]]; then
    TRAINDATA=multi30k
fi
if [[ "${TRAINDATA}" == "gmnmt" ]]; then
    TRAIN_PREFIX=gm.
else
    TRAIN_PREFIX=""
fi
TESTDATA=${3}
# options: multi30k-test2016, multi30k-test2017, mscoco-test2017
#          default: multi30k-test2016
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
DATA_CONFIG=${DATA_CONFIG}/${TRAINDATA}.parsed.${TESTDATA}.config.json
MODEL_CONFIG=${PROJECT_DIR}/configurations/model_configurations
MODEL_CONFIG=${MODEL_CONFIG}/transformer-config.json

SUFFIX=${TRAIN_PREFIX}nobpe.dp0.2.r${RANDOM_SEED}.${SUFFIX}
OUT_DIR_BASE=${PROJECT_DIR}/experiment/models/transformer
#OUT_DIR_BASE=${PROJECT_DIR}/experiment/models/transformer.en2fr
#OUT_DIR_BASE=${PROJECT_DIR}/experiment/models/transformer.en2cs
OUT_DIR=${OUT_DIR_BASE}/transformer.self_designed.${SUFFIX}
mkdir -p ${PROJECT_DIR}/experiment/records/${OUT_DIR_BASE##*/}
cp $0 ${OUT_DIR}/
nohup python ${PROJECT_DIR}/infer.py \
    --config_file=${MODEL_CONFIG} \
    --dataset_config_file=${DATA_CONFIG} \
    --translation_filename=${TESTDATA}_translations.txt \
    --out_dir=${OUT_DIR} \
    --additional=${ADDITIONAL} \
    --infer_record_file=${PROJECT_DIR}/experiment/records/transformer/${OUT_DIR_BASE##*/}.${TRAINDATA}test${TESTDATA}.txt \
    > ${OUT_DIR}/infer${TESTDATA}log.txt 2>&1 &
