# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:train_procedure.py
@time:2020-08-28 
@desc:
'''
import argparse
import numpy as np

from utils import utils, config
from datasets.dataset_manager import DatasetManager as DatasetManager
from torchtext_datasets.torchtext_dataset_manager import DatasetManager as TTDatasetManager

# from procedures.eval_procedure import eval_procedure
# from procedures.infer_procedure import infer_procedure

from utils.config import DATA_PART, MODE


def train_procedure(FLAGS):
    TRAINING_MANAGER_FUNCTION = FLAGS.training_manager_function
    APPLY_TORCHTEXT = FLAGS.apply_torchtext

    OUT_DIR = FLAGS.out_dir  # specify in shell script
    CHECKPOINT_NAME = FLAGS.checkpoint_name  # default
    INITIAL_FUNC_NAME = FLAGS.initial_func_name  # specify in shell script

    OPTIMIZER = FLAGS.optimizer
    OPTIMIZE_DELAY = FLAGS.optimize_delay
    LEARNING_RATE = FLAGS.learning_rate

    MAX_STEPS_WITHOUT_CHANGE = FLAGS.max_steps_without_change
    DECAY_FACTOR = FLAGS.decay_factor
    START_DECAY_STEP = FLAGS.start_decay_step
    DECAY_STEP_SIZE = FLAGS.decay_step_size

    EPOCH = FLAGS.epoch
    STEPS_PER_INTERNAL_EVAL = FLAGS.steps_per_internal_eval
    CHECKPOINT_KEEP_NUMBER = FLAGS.checkpoint_keep_number
    NUM_CHECKPOINT_PER_AVERAGE = FLAGS.num_checkpoint_per_average
    AVERAGE_CHECKPOINT = FLAGS.average_checkpoint
    STEPS_PER_INFER = FLAGS.steps_per_infer
    STEPS_PER_EXTERNAL_EVAL = FLAGS.steps_per_external_eval
    AVERAGED_MODEL_CHECKPOINT = FLAGS.averaged_model_checkpoint_name
    FINAL_TEST = FLAGS.final_test

    # create dataset manager
    if APPLY_TORCHTEXT:
        dm = TTDatasetManager(FLAGS, which_part=[DATA_PART.Train, DATA_PART.Val, DATA_PART.Test])
    else:
        dm = DatasetManager(FLAGS,
                            which_part=[DATA_PART.Train, DATA_PART.Val, DATA_PART.Test])

    # create training manager
    from model_managers.rnn_seq2seq_training_manager import RNNSeq2SeqTrainingManager
    from model_managers.transformer_training_manager import TransformerTrainingManager
    from model_managers.transformer_imagine_training_manager import TransformerImagineTrainingManager
    from model_managers.rnn_token_imagine_manager import RNNTokenImagineManager
    tm = eval(TRAINING_MANAGER_FUNCTION)(MODE.Train, FLAGS)  # training manager
    model = tm.create_model()
    tm.freeze_parameters(model)
    model = tm.parallel_model(model)

    criterions = tm.create_criterions(MODE.Train)
    # criterions = tm.criterions
    device = tm.device
    summary_writer = tm.summary_writer
    # model, criterions = tm.to_device(model, criterions, device)
    tm.to_device(model, criterions, device)
    # tm.model = model
    # tm.criterions = criterions
    optimizer = tm.create_optimizer(OPTIMIZER, LEARNING_RATE, model)

    print(model)
    print('avaliable device:', device)
    #
    loaded = utils.load_or_create_model(model,
                                        OUT_DIR,
                                        checkpoints_filename=CHECKPOINT_NAME,
                                        initial_func_name=INITIAL_FUNC_NAME)
    if not loaded:
        # which means that the model was created but not loaded.
        tm.load_pretrained_model(model, dm)

    # start training procedure
    print('start training:')
    stoper = tm.set_stoper(MAX_STEPS_WITHOUT_CHANGE)
    scheduler = tm.set_scheduler(optimizer, step_size=DECAY_STEP_SIZE, gamma=DECAY_FACTOR)
    dataloader = dm.get_dataloader(DATA_PART.Train)
    tm.now_as_start_time(MODE.Train)
    for current_epoch in range(EPOCH):
        for i, packed_data in enumerate(dataloader):
            if APPLY_TORCHTEXT:
                packed_data = dm.pack_data(packed_data, DATA_PART.Train)
            model.train()
            optimizer.zero_grad()

            batch_info = tm.loss_forward(current_epoch, i, packed_data, model, criterions=criterions, device=device)
            loss = batch_info['loss']
            assert not np.isnan(loss.item())
            tm.backward(batch_info, model)
            optimizer.step()

            tm.training_one_more_step(batch_info)
            # print(running_loss)
            if tm.training_total_step > 0 and \
                    tm.training_total_step % (STEPS_PER_INTERNAL_EVAL * OPTIMIZE_DELAY) == 0:
                tm.report_internal_evaluation(model, current_epoch, i, summary_writer)
                utils.save_model(model, OUT_DIR, CHECKPOINT_NAME, tm.training_total_step,
                                 keep_number=CHECKPOINT_KEEP_NUMBER)
                # if tm.training_total_step % (
                #         STEPS_PER_INTERNAL_EVAL * NUM_CHECKPOINT_PER_AVERAGE) == 0 and AVERAGE_CHECKPOINT:
                #     utils.average_lastN(model,
                #                         OUT_DIR,
                #                         N=NUM_CHECKPOINT_PER_AVERAGE,
                #                         averaged_checkpoints_filename=AVERAGED_MODEL_CHECKPOINT)
            if STEPS_PER_INFER > 0 and \
                    tm.training_total_step > 0 and \
                    tm.training_total_step % (STEPS_PER_INFER * OPTIMIZE_DELAY) == 0:
                tm.test(FLAGS,
                        dataset_part=DATA_PART.Test,
                        training_manager=tm,
                        dataset_manager=dm,
                        checkpoint_name=None,
                        global_step=tm.training_total_step)
            if STEPS_PER_EXTERNAL_EVAL > 0 and \
                    tm.training_total_step > 0 and \
                    tm.training_total_step % (STEPS_PER_EXTERNAL_EVAL * OPTIMIZE_DELAY) == 0:
                stop_signal = tm.validate(FLAGS,
                                          dataset_part=DATA_PART.Val,
                                          training_manager=tm,
                                          dataset_manager=dm,
                                          checkpoint_name=None,
                                          global_step=tm.training_total_step)
                if stoper.if_stoped(stop_signal):
                    tm.stop_training()
                    break
        if scheduler is not None and current_epoch >= START_DECAY_STEP:
            scheduler.step()
        if tm.is_training_stopped():
            break

    ckpts_name = None
    if AVERAGE_CHECKPOINT:
        ckpts_name = AVERAGED_MODEL_CHECKPOINT
        utils.average_lastN(model,
                            OUT_DIR,
                            N=NUM_CHECKPOINT_PER_AVERAGE,
                            averaged_checkpoints_filename=AVERAGED_MODEL_CHECKPOINT)
    if FINAL_TEST:
        tm.test(FLAGS,
            dataset_part=DATA_PART.Test,
            training_manager=tm,
            dataset_manager=dm,
            checkpoint_name=ckpts_name,
            global_step=tm.training_total_step)
    summary_writer.close()


def main(FLAGS):
    train_procedure(FLAGS)


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    config.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    main(FLAGS)
