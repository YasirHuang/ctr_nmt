# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:train_procedure.py
@time:2020-08-28 
@desc:
'''
import argparse

import torch

from utils import utils, config
from datasets.dataset_manager import DatasetManager as DatasetManager
from torchtext_datasets.torchtext_dataset_manager import DatasetManager as TTDatasetManager

from utils.config import DATA_PART, MODE


def eval_procedure(FLAGS,
                   dataset_part=DATA_PART.Val,
                   training_manager=None,
                   dataset_manager=None,
                   checkpoint_name=None,
                   global_step=0):
    APPLY_TORCHTEXT = FLAGS.apply_torchtext
    TRAINING_MANAGER_FUNCTION = FLAGS.training_manager_function

    OUT_DIR = FLAGS.out_dir
    CHECKPOINT_NAME = checkpoint_name if checkpoint_name else FLAGS.checkpoint_name

    if not dataset_manager:
        # dm = DatasetManager(FLAGS,
        #                     which_part=dataset_part)
        if APPLY_TORCHTEXT:
            dm = TTDatasetManager(FLAGS, which_part=dataset_part)
        else:
            dm = DatasetManager(FLAGS, which_part=dataset_part)
    else:
        dm = dataset_manager

    if training_manager is None:
        from model_managers.rnn_seq2seq_training_manager import RNNSeq2SeqTrainingManager
        from model_managers.transformer_training_manager import TransformerTrainingManager
        from model_managers.transformer_imagine_training_manager import TransformerImagineTrainingManager
        from model_managers.rnn_token_imagine_manager import RNNTokenImagineManager
        tm = eval(TRAINING_MANAGER_FUNCTION)(MODE.Eval, FLAGS)  # training manager
        model = tm.create_model()
        criterions = tm.criterions
        device = tm.device
        tm.to_device(model, criterions, device)
        utils.load_model(model, OUT_DIR, CHECKPOINT_NAME)
    else:
        tm = training_manager
        model = tm.model
        criterions = tm.criterions
        device = tm.device

    # model, criterions = tm.to_device(model, criterions, device)
    # tm.model = model
    # tm.criterions = criterions
    # loaded = utils.load_or_create_model(tm.model,
    #                                     OUT_DIR,
    #                                     checkpoints_filename=CHECKPOINT_NAME,
    #                                     initial_func_name=INITIAL_FUNC_NAME)

    # start training procedure
    print('start evaluation:')
    dataloader = dm.get_dataloader(dataset_part)
    tm.now_as_start_time(MODE.Eval)
    with torch.no_grad():
        model.eval()
        for i, packed_data in enumerate(dataloader):
            if APPLY_TORCHTEXT:
                packed_data = dm.pack_data(packed_data, dataset_part)
            batch_info = tm.loss_forward(0, 0,
                packed_data,
                                   model,
                                   criterions=criterions,
                                   device=device)
            tm.evaluating_one_more_step(batch_info)

    total_loss = tm.report_external_evaluation(model, global_step=global_step)
    torch.cuda.empty_cache()
    return total_loss


def evaluation(FLAGS):
    eval_procedure(FLAGS)


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    config.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    evaluation(FLAGS)
