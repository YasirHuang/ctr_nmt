# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:infer_procedure.py
@time:2020-08-31 
@desc:
'''

import os
import argparse

import torch

from utils import utils, config, nlgeval_utils
from datasets.dataset_manager import DatasetManager as DatasetManager
from torchtext_datasets.torchtext_dataset_manager import DatasetManager as TTDatasetManager

from utils.config import DATA_PART, MODE
from utils.config import datapart_tostr


def infer_procedure(FLAGS,
                    dataset_part=DATA_PART.Test,
                    training_manager=None,
                    dataset_manager=None,
                    checkpoint_name=None,
                    global_step=None):
    APPLY_TORCHTEXT = FLAGS.apply_torchtext
    BEAM_WIDTH = FLAGS.beam_width
    MAX_DECODE_STEP = FLAGS.max_decode_step  # default 50
    MAX_DECODE_STEP_RATIO = FLAGS.max_decode_step_ratio  # default 2.0
    BPE_DELIMITER = FLAGS.bpe_delimiter  # default @@

    TRAINING_MANAGER_FUNCTION = FLAGS.training_manager_function

    OUT_DIR = FLAGS.out_dir
    TRANSLATION_FILE = FLAGS.translation_filename  # default "translations.txt"
    CHECKPOINT_NAME = checkpoint_name if checkpoint_name else FLAGS.checkpoint_name
    INITIAL_FUNC_NAME = FLAGS.initial_func_name
    RECORD_FILE = FLAGS.infer_record_file
    STORE_ATTENTION = FLAGS.store_attention

    BLEU_CMD = FLAGS.bleu_cmd  # default 'perl /home/xhuang/work/tools/multi-bleu.perl'
    METEOR_CMD = FLAGS.meteor_cmd
    METEOR_JAR = FLAGS.meteor_jar

    if not dataset_manager:
        # dm = DatasetManager(FLAGS,
        #                     which_part=dataset_part)
        if APPLY_TORCHTEXT:
            dm = TTDatasetManager(FLAGS, which_part=dataset_part)
        else:
            dm = DatasetManager(FLAGS,
                                which_part=dataset_part)
    else:
        dm = dataset_manager

    if not training_manager:
        from model_managers.rnn_seq2seq_training_manager import RNNSeq2SeqTrainingManager
        from model_managers.transformer_training_manager import TransformerTrainingManager
        from model_managers.transformer_imagine_training_manager import TransformerImagineTrainingManager
        from model_managers.rnn_token_imagine_manager import RNNTokenImagineManager
        tm = eval(TRAINING_MANAGER_FUNCTION)(MODE.Infer, FLAGS)  # training manager
        model = tm.create_model()
        device = tm.device
        tm.to_device(model, device=device)
        utils.load_model(model, OUT_DIR, CHECKPOINT_NAME)
    else:
        tm = training_manager
        model = tm.model
        device = tm.device

    if STORE_ATTENTION and hasattr(tm, 'add_hooks'):
        tm.add_hooks()
    # if not loaded:
    #     # which means that the model was created but not loaded.
    #     tm.load_pretrained_model(model, dm)

    # start inference procedure
    print('start inference:')
    dataloader = dm.get_dataloader(dataset_part)
    tm.now_as_start_time(MODE.Infer)
    sentences = []

    with torch.no_grad():
        model.eval()
        for i, packed_data in enumerate(dataloader):
            if APPLY_TORCHTEXT:
                packed_data = dm.pack_data(packed_data, dataset_part)
            seqs, att_scores = tm.hypo_forward(packed_data,
                                               model,
                                               dm.tgt_word_to_id_dict,
                                               device=device,
                                               beam_width=BEAM_WIDTH,
                                               max_decode_step=MAX_DECODE_STEP,
                                               max_decode_step_ratio=MAX_DECODE_STEP_RATIO,
                                               src_sos=dm.src_sos,
                                               tgt_sos=dm.tgt_sos,
                                               eos=dm.eos,
                                               unk=dm.unk)
            tm.testing_one_more_step()

            seqs = seqs.view(-1, BEAM_WIDTH, seqs.shape[-1])
            for seq in seqs[:, 0, :]:
                sent = []
                for wid in seq[1:]:
                    word = dm.tgt_id_to_word_dict[wid.item()].encode()
                    sent.append(word)
                sentences.append(utils.get_translation(sent, dm.eos, BPE_DELIMITER))

    trans_file = os.path.join(OUT_DIR, "%s_%s" %
                              (datapart_tostr(dataset_part), TRANSLATION_FILE))
    with open(trans_file, 'w') as fp:
        for sent in sentences:
            fp.write((sent + b'\n').decode(encoding='utf-8'))
    ref_file = dm.dataset_profile.get_reference_file(dataset_part, dm.tgt)
    # result = os.system('%s %s < %s' % (cmd, ref_file, trans_file))
    bleu_score = utils.moses_bleu(BLEU_CMD, ref_file, trans_file)
    meteor_score = utils.meteor_score(METEOR_CMD,
                                      ref_file, trans_file,
                                      dm.tgt, METEOR_JAR)
    metric_results = {"Bleu_4": bleu_score, "METEOR": meteor_score}
    tm.report_inference(model,
                        bleu_score,
                        name="infer %s bleu_score" % datapart_tostr(dataset_part),
                        summary_writer=None,
                        global_step=global_step)
    tm.report_inference(model,
                        meteor_score,
                        name="infer %s meteor_score" % datapart_tostr(dataset_part),
                        summary_writer=None,
                        global_step=global_step)
    if RECORD_FILE is not None:
        record_line = utils.format_results('infer',
                                           OUT_DIR,
                                           FLAGS.dataset_config_file,
                                           metric_results)
        utils.record_results(RECORD_FILE, record_line)
    torch.cuda.empty_cache()
    if STORE_ATTENTION and hasattr(tm, 'add_hooks') and hasattr(tm, 'remove_hooks'):
        tm.remove_hooks(beam_width=BEAM_WIDTH, sentences=sentences, out_dir=OUT_DIR)
    return bleu_score


def infer(FLAGS):
    infer_procedure(FLAGS)


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    config.add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    infer(FLAGS)
