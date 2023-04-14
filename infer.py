# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:infer.py
@time:2020-01-08 
@desc:
'''
import sys
import json
import argparse
from utils import utils, config
from procedures.infer_procedure import infer as infer_procedure

def main(FLAGS):
    external_config = config.merge_configurations(
        FLAGS.baseline_config_file,
        FLAGS.config_file,
        sys.argv,
        nmt_parser
    )
    FLAGS.__dict__.update(external_config)
    if FLAGS.additional is not None and len(FLAGS.additional) > 0:
        additional = json.loads(FLAGS.additional)
        FLAGS.__dict__.update(additional)
    utils.check_hparams(FLAGS)
    utils.print_hparams(FLAGS)
    utils.copy_hparams_to_outdir(FLAGS)
    infer_procedure(FLAGS)
    # eval(FLAGS.infer_function)(FLAGS)


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    config.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    main(FLAGS)