#!/usr/bin/env bash
# e.g.: ./infer_remmt.sh 0 src i i k multi30k multi30k-test2016 1234 0
#       ./infer_remmt.sh 0 src i i k d d 1234 0
PROJECT_DIR=.
export CUDA_VISIBLE_DEVICES=${1}
## freedom parameters
#IMAGINE_TO_SRC=True
#SHARE_DECODER=False
#SHARE_OPTIMIZER=True
#SHARE_EMBEDDING=False
#KEY_WORD_OWN_IMAGE=True
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
if [[ "${4}" == "s" ]]; then
    SHARE_EMBEDDING=True
else
    SHARE_EMBEDDING=False
fi
if [[ "${5}" == "k" ]]; then
    KEY_WORD_OWN_IMAGE=True
else
    KEY_WORD_OWN_IMAGE=False
fi
TRAINDATA=${6}
# options: gmnmt, multi30k. default: multi30k
if [[ "${TRAINDATA}" != "gmnmt" ]]; then
    TRAINDATA=multi30k
fi
if [[ "${TRAINDATA}" == "gmnmt" ]]; then
    TRAIN_PREFIX=gm.
else
    TRAIN_PREFIX=""
fi
TESTDATA=${7}
# options: multi30k-test2016, multi30k-test2017, mscoco-test2017
#          default: multi30k-test2016
if [[ "${TESTDATA}" != "multi30k-test2017" && "${TESTDATA}" != "mscoco-test2017" ]]; then
    TESTDATA=multi30k-test2016
fi
RANDOM_SEED=${8}
if [[ "${RANDOM_SEED}" == "" ]]; then
    RANDOM_SEED=1234
fi
SUFFIX=${9}
if [[ "${SUFFIX}" == "" ]]; then
    SUFFIX=0
fi
ADDITIONAL=${10}

###########################################
# constrains:
if [[ "${IMAGINE_TO_SRC}" == "False" ]]; then
#   1) if to_tgt then share_decoder
    SHARE_DECODER=True
elif [[ "${IMAGINE_TO_SRC}" == "True" && "${SHARE_DECODER}" == "True" ]]; then
#   2) if to_src and share_decoder then share_embedding
    SHARE_EMBEDDING=True
fi
if [[ "${SHARE_DECODER}" == "True" ]]; then
#   3) if share_decoder then share_opt
#       if to_src then share_embedding
    SHARE_OPTIMIZER=True
    if [[ "${IMAGINE_TO_SRC}" == "True" ]]; then
        SHARE_EMBEDDING=True
    fi
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
EMB=semb
if [[ "${SHARE_EMBEDDING}" == "False"  ]]; then
    EMB=iemb
fi
WORD=key
if [[ "${KEY_WORD_OWN_IMAGE}" == "False"  ]]; then
    WORD=all
fi

DATA_CONFIG=${PROJECT_DIR}/configurations/data_configurations
DATA_CONFIG=${DATA_CONFIG}/${TRAINDATA}.parsed.${TESTDATA}.config.json
MODEL_CONFIG=${PROJECT_DIR}/configurations/model_configurations
MODEL_CONFIG=${MODEL_CONFIG}/std-rnn-token-imagine-config.json

SUFFIX=${TRAIN_PREFIX}${TO}.${DEC}.sopt.${EMB}.${WORD}.initdec.r${RANDOM_SEED}.${SUFFIX}
#OUT_DIR_BASE=${PROJECT_DIR}/experiment/models/remmt
#OUT_DIR_BASE=${PROJECT_DIR}/experiment/models/remmt.en2fr
OUT_DIR_BASE=${PROJECT_DIR}/experiment/models/remmt.en2cs
OUT_DIR=${OUT_DIR_BASE}/luong_lstm.standard_uniform.${SUFFIX}

mkdir -p ${PROJECT_DIR}/experiment/records/${OUT_DIR_BASE##*/}
cp $0 ${OUT_DIR}/
nohup python ${PROJECT_DIR}/infer.py \
    --config_file=${MODEL_CONFIG} \
    --dataset_config_file=${DATA_CONFIG} \
    --initial_func_name=standard_uniform \
    --individual_start_token=${INDIVIDUAL_START_TOKEN} \
    --imagine_to_src=${IMAGINE_TO_SRC} \
    --share_decoder=${SHARE_DECODER} \
    --share_optimizer=True \
    --share_embedding=${SHARE_EMBEDDING} \
    --share_vocab=${SHARE_EMBEDDING} \
    --key_word_own_image=${KEY_WORD_OWN_IMAGE} \
    --translation_filename=${TESTDATA}_translations.txt \
    --out_dir=${OUT_DIR} \
    --additional=${ADDITIONAL} \
    --infer_record_file=${PROJECT_DIR}/experiment/records/${OUT_DIR_BASE##*/}/${OUT_DIR_BASE##*/}.${TRAINDATA}test${TESTDATA}.txt \
    > ${OUT_DIR}/infer${TESTDATA}log.txt 2>&1 &