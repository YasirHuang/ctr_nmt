# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:train_procedure.py
@time:2020-01-08 
@desc:
'''
import sys
import torch
import json
import random
import argparse
import numpy as np
from utils import utils, config
from procedures.train_procedure import train_procedure

# torch.backends.cudnn.benchmark = True

def set_seeds(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CustomNamespace(argparse.Namespace):

    def __init__(self, **kwargs):
        super(CustomNamespace, self).__init__(**kwargs)
        self.param_register = list()


def main():
    nmt_parser = argparse.ArgumentParser()
    # add default arguments
    config.add_arguments(nmt_parser)
    # parse system input arguments
    # the first time parse_known_args to get config_file
    FLAGS, unparsed = nmt_parser.parse_known_args()
    # TODO: add baseline model configuration
    # parse arguments from json file
    external_config = config.merge_configurations(
        FLAGS.baseline_config_file,
        FLAGS.config_file,
        sys.argv,
        nmt_parser
    )
    FLAGS.__dict__.update(external_config)
    if FLAGS.additional is not None and len(FLAGS.additional) > 0:
        print(FLAGS.additional)
        additional = json.loads(FLAGS.additional)
        FLAGS.__dict__.update(additional)

    # the second time parse_known_args to get other configs
    # so that the config from shell has higher priority than config_file
    # additioal > shell > config_file > default
    # FLAGS, unparsed = nmt_parser.parse_known_args()

    utils.check_hparams(FLAGS)
    utils.print_hparams(FLAGS)
    utils.copy_hparams_to_outdir(FLAGS)
    set_seeds(FLAGS.random_seed)
    train_procedure(FLAGS)

if __name__ == "__main__":
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    main()