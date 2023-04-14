#!/usr/bin/env bash
# e.g.: ./infer_temmt.sh 0 src i k multi30k multi30k-test2016 1234 0
#       ./infer_temmt.sh 0 src i k d d 1234 0
PROJECT_DIR=.
export CUDA_VISIBLE_DEVICES=${1};echo "CUDA_VISIBLE_DEVICES=${1}"
## freedom parameters
#IMAGINE_TO_SRC=True
#SHARE_DECODER=False
#SHARE_OPTIMIZER=True
##SHARE_EMBEDDING=False
#KEY_WORD_OWN_IMAGE=False
if [[ "${2}" == "s" || "${2}" == "src" ]]; then
    IMAGINE_TO_SRC=True
else
    IMAGINE_TO_SRC=False
fi
if [[ "${3}" == "s" ]]; then
    SHARE_DECODER=True
else
    SHARE_DECODER=False
fi
if [[ "${4}" == "k" ]]; then
    KEY_WORD_OWN_IMAGE=True
else
    KEY_WORD_OWN_IMAGE=False
fi
TRAINDATA=${5}
# options: gmnmt, multi30k. default: multi30k
if [[ "${TRAINDATA}" != "gmnmt" ]]; then
    TRAINDATA=multi30k
fi
if [[ "${TRAINDATA}" == "gmnmt" ]]; then
    TRAIN_PREFIX=gm.
else
    TRAIN_PREFIX=""
fi
TESTDATA=${6}
# options: multi30k-test2016, multi30k-test2017, mscoco-test2017
#          default: multi30k-test2016
if [[ "${TESTDATA}" != "multi30k-test2017" && "${TESTDATA}" != "mscoco-test2017" ]]; then
    TESTDATA=multi30k-test2016
fi
RANDOM_SEED=${7}
if [[ "${RANDOM_SEED}" == "" ]]; then
    RANDOM_SEED=1234
fi
SUFFIX=${8}
if [[ "${SUFFIX}" == "" ]]; then
    SUFFIX=0
fi

ADDITIONAL=${9}
###########################################
# constrains:
if [[ "${IMAGINE_TO_SRC}" == "False" ]]; then
    SHARE_DECODER=True
fi
if [[ "${IMAGINE_TO_SRC}" == "True" && "${SHARE_DECODER}" == "True" ]]; then
    INDIVIDUAL_START_TOKEN=True
else
    INDIVIDUAL_START_TOKEN=False
fi

# configurations to output file string
TO=tosrc
if [[ "${IMAGINE_TO_SRC}" == "False"  ]]; then
    TO=totgt
fi
DEC=sdec
if [[ "${SHARE_DECODER}" == "False"  ]]; then
    DEC=idec
fi
WORD=key
if [[ "${KEY_WORD_OWN_IMAGE}" == "False"  ]]; then
    WORD=all
fi

DATA_CONFIG=${PROJECT_DIR}/configurations/data_configurations
DATA_CONFIG=${DATA_CONFIG}/${TRAINDATA}.parsed.${TESTDATA}.config.json
MODEL_CONFIG=${PROJECT_DIR}/configurations/model_configurations
MODEL_CONFIG=${MODEL_CONFIG}/transformer-imagine-config.json

SUFFIX=${TRAIN_PREFIX}${TO}.${DEC}.sopt.${WORD}.r${RANDOM_SEED}.${SUFFIX}
#OUT_DIR_BASE=${PROJECT_DIR}/experiment/models/transformer_imagine
OUT_DIR_BASE=${PROJECT_DIR}/experiment/models/temmt.en2fr
#OUT_DIR_BASE=${PROJECT_DIR}/experiment/models/temmt.en2cs
OUT_DIR=${OUT_DIR_BASE}/transformer.self_designed.${SUFFIX}

mkdir -p ${PROJECT_DIR}/experiment/records/${OUT_DIR_BASE##*/}
cp $0 ${OUT_DIR}/
nohup python ${PROJECT_DIR}/infer.py \
    --config_file=${MODEL_CONFIG} \
    --dataset_config_file=${DATA_CONFIG} \
    --out_dir=${OUT_DIR} \
    --gpu_id=${CUDA_VISIBLE_DEVICES} \
    --individual_start_token=${INDIVIDUAL_START_TOKEN} \
    --imagine_to_src=${IMAGINE_TO_SRC} \
    --share_decoder=${SHARE_DECODER} \
    --share_optimizer=True \
    --key_word_own_image=${KEY_WORD_OWN_IMAGE} \
    --translation_filename=${TESTDATA}_translations.txt \
    --additional=${ADDITIONAL} \
    --infer_record_file=${PROJECT_DIR}/experiment/records/${OUT_DIR_BASE##*/}/${OUT_DIR_BASE##*/}.${TRAINDATA}test${TESTDATA}.txt \
    > ${OUT_DIR}/infer${TESTDATA}log.txt 2>&1 &