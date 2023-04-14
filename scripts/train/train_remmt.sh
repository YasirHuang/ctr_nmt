#!/usr/bin/env bash
# e.g.: ./train_remmt.sh 0 src i i k multi30k multi30k-test2016 1234 0
#       ./train_remmt.sh 0 src i i k d d 1234 0
PROJECT_DIR=.
export CUDA_VISIBLE_DEVICES=${1};echo "CUDA_VISIBLE_DEVICES=${1}"
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
# options: gmnmt, multi30k.parsed default: multi30k.parsed
if [[ "${TRAINDATA}" != "gmnmt" ]]; then
    TRAINDATA=multi30k.parsed
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
DATA_CONFIG=${DATA_CONFIG}/${TRAINDATA}.${TESTDATA}.config.json
MODEL_CONFIG=${PROJECT_DIR}/configurations/model_configurations
MODEL_CONFIG=${MODEL_CONFIG}/std-rnn-token-imagine-config.json

SUFFIX=${TRAIN_PREFIX}${TO}.${DEC}.sopt.${EMB}.${WORD}.initdec.r${RANDOM_SEED}.${SUFFIX}
#OUT_DIR=${PROJECT_DIR}/experiment/models/rnn_token_imagine
OUT_DIR=${PROJECT_DIR}/experiment/models/remmt.en2cs
OUT_DIR=${OUT_DIR}/luong_lstm.standard_uniform.${SUFFIX}

mkdir -p ${OUT_DIR}
cp $0 ${OUT_DIR}/
echo "$0 $*" > ${OUT_DIR}/cmdlog.txt
nohup python ${PROJECT_DIR}/train.py \
    --config_file=${MODEL_CONFIG} \
    --dataset_config_file=${DATA_CONFIG} \
    --out_dir=${OUT_DIR} \
    --gpu_id=${CUDA_VISIBLE_DEVICES} \
    --initial_func_name=standard_uniform \
    --random_seed=${RANDOM_SEED} \
    --individual_start_token=${INDIVIDUAL_START_TOKEN} \
    --imagine_to_src=${IMAGINE_TO_SRC} \
    --share_decoder=${SHARE_DECODER} \
    --share_optimizer=True \
    --share_embedding=${SHARE_EMBEDDING} \
    --share_vocab=${SHARE_EMBEDDING} \
    --key_word_own_image=${KEY_WORD_OWN_IMAGE} \
    --additional=${ADDITIONAL} \
    > ${OUT_DIR}/log.txt 2>&1 &


#                                            modules to be shared
#                                       decoder   optmizer  embedding
#├── tosrc
#│   ├── seperate_decoder
#│   │   ├── seperate_opt
#│   │   │   ├── seperate_embedding     F         F         F
#│   │   │   └── share_embedding        F         F         T
#│   │   └── share_opt
#│   │       ├── seperate_embedding     F         T         F
#│   │       └── share_embedding        F         T         T
#│   └── share_decoder
#│       └── share_opt
#│           └── share_embedding        T         T         T
#└── totgt
#    └── share_decoder
#        └── share_opt
#            ├── seperate_embedding     T         T         F
#            └── share_embedding        T         T         T
# constrains:
#   1) if to_tgt then share_decoder
#   2) if to_src and share_decoder then share_embedding
#   3) if share_decoder then share_opt
#       if to_src then share_embedding
