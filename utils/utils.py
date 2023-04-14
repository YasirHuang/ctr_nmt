# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:utils.py
@time:2019-12-10 
@desc:
'''

import argparse
import os, shutil
import subprocess
import re
import sys
import json
import numpy as np
import collections
import torch
import torch.nn as nn

def safe_division(numerator, denominator):
    if denominator != 0:
        return numerator / denominator
    if numerator == 0:
        return 0.0
    elif numerator > 0:
        return float('inf')
    else:
        return float('-inf')

def moses_bleu(command_line, referece_filename, hypothesis_filename):
    cmd = "%s %s < %s" % (command_line, referece_filename, hypothesis_filename)
    # subprocess
    bleu_output = subprocess.check_output(cmd, shell=True)

    # extract BLEU score
    m = re.search("BLEU = (.+?),", bleu_output.decode('utf-8'))
    bleu_score = float(m.group(1))
    return bleu_score

def meteor_score(command_line, referece_filename, hypothesis_filename,
                 target_language, meteor_jar):
    cmd = command_line.format(meteor_jar=meteor_jar,
                              hypo=hypothesis_filename,
                              ref=referece_filename,
                              lang=target_language)
    meteor_output = subprocess.check_output(cmd, shell=True)
    score_line = meteor_output.decode("utf-8").strip().split()[-1]
    return float(score_line)*100.0

def print_out(s, f=None, new_line=True):
    '''
    print log s, if file f is provided, print s to f
    :param s: message to print
    :param f: file to output, default stdout
    :param new_line: if starting a new line, default to true
    :return:
    '''
    """Similar to print but with support to flush and output to a file."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def text_preprocess(file_name, word_to_id_dict, sos='<sos>', eos='<eos>', unk='<unk>'):
    with open(file_name, 'r') as fp:
        lines = fp.readlines()
        lines = [[word2id(w, word_to_id_dict, unk) for w in line.strip().split()] for line in lines]

    sos_id = word2id(sos, word_to_id_dict, unk)
    eos_id = word2id(eos, word_to_id_dict, unk)

    sos_lines = []
    for l in lines:
        s_l = [sos_id]
        s_l.extend(l)
        sos_lines.append(s_l)

    eos_lines = []
    for l in lines:
        e_l = l
        e_l.append(eos_id)
        eos_lines.append(e_l)

    seq_lens = [len(l) for l in sos_lines]
    return np.array(sos_lines), np.array(eos_lines), np.array(seq_lens)


def padding_batch(data, padding_data, seq_lens):
    # data: [batch_size, seq_lens, data_size]
    # padded_data: [data_size]
    # seq_lens: [batch_size]
    max_seq_len = np.max(seq_lens)

    padded_data = []
    for seq, seq_len in zip(data, seq_lens):
        seq = seq[:seq_len]
        seq.extend([padding_data] * (max_seq_len - seq_len))
        assert len(seq) == max_seq_len, \
            "seq_len=%d, max_seq_len=%d, seq_len=%d, %s, %s" % (len(seq), max_seq_len, seq_len, str(seq_lens), str(seq))

        padded_data.append(seq)
        # seq.extend([padded_data] * (max_seq_len - seq_len))
    return np.array(padded_data)


def word2id(word, word_to_id_dict, unk):
    if word.strip() in word_to_id_dict:
        return word_to_id_dict[word]
    else:
        return word_to_id_dict[unk]


def build_word2iddict(dict_file,
                      initial_dict=None,
                      sos='<sos>',
                      eos='<eos>',
                      unk='<unk>',
                      pad='<pad>',
                      special_tokens=None):
    with open(dict_file, 'r') as fp:
        lines = fp.readlines()

    if initial_dict:
        word_to_id_dict = initial_dict
    else:
        word_to_id_dict = {pad: 0, sos: 1, eos: 2, unk: 3}

    if special_tokens is not None:
        assert isinstance(special_tokens, (list, tuple, str))
        if isinstance(special_tokens, str):
            special_tokens = [special_tokens]
        for st in special_tokens:
            if not st in word_to_id_dict:
                word_to_id_dict[st] = len(word_to_id_dict)

    initial_dict_len = len(word_to_id_dict)
    for i, w in enumerate(lines):
        if w.strip() in word_to_id_dict:
            continue
        word_to_id_dict[w.strip()] = i + initial_dict_len
    return word_to_id_dict, len(word_to_id_dict)


def build_id2worddict(dict_file,
                      word_to_id_dict=None,
                      initial_dict=None,
                      sos='<sos>',
                      eos='<eos>',
                      unk='<unk>',
                      pad='<pad>',
                      special_tokens=None):
    id_to_word_dict = {}
    if word_to_id_dict:
        for word in word_to_id_dict:
            id_to_word_dict[word_to_id_dict[word]] = word
    else:
        with open(dict_file, 'r') as fp:
            lines = fp.readlines()
        if initial_dict:
            id_to_word_dict = initial_dict
        else:
            id_to_word_dict[0] = pad
            id_to_word_dict[1] = sos
            id_to_word_dict[2] = eos
            id_to_word_dict[3] = unk

        if special_tokens is not None:
            assert isinstance(special_tokens, (list, tuple, str))
            if isinstance(special_tokens, str):
                special_tokens = [special_tokens]
            for st in special_tokens:
                if not st in id_to_word_dict:
                    id_to_word_dict[len(id_to_word_dict)] = st

        initial_dict_len = len(id_to_word_dict)
        for i, w in enumerate(lines):
            id_to_word_dict[i + initial_dict_len] = w.strip()
    return id_to_word_dict, len(word_to_id_dict)


def format_text(words):
    """Convert a sequence words into sentence."""
    if (not hasattr(words, "__len__") and  # for numpy array
            not isinstance(words, collections.Iterable)):
        words = [words]
    return b" ".join(words)


def format_bpe_text(symbols, delimiter=b"@@"):
    """Convert a sequence of bpe words into sentence."""
    words = []
    word = b""
    if isinstance(symbols, str):
        symbols = symbols.encode()
    delimiter_len = len(delimiter)
    for symbol in symbols:
        if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
            word += symbol[:-delimiter_len]
        else:  # end of a word
            word += symbol
            words.append(word)
            word = b""
    return b" ".join(words)


def get_translation(nmt_output, tgt_eos, bpe_delimiter):
    """Given batch decoding outputs, select a sentence and turn to text."""
    if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
    if bpe_delimiter: bpe_delimiter = bpe_delimiter.encode("utf-8")
    # Select a sentence
    output = nmt_output

    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]

    # if '.' in output:
    #     output = output[:output.index('.') + 1]

    if not bpe_delimiter:
        translation = format_text(output)
    else:  # BPE
        translation = format_bpe_text(output, delimiter=bpe_delimiter)

    return translation


def check_hparams(FLAGS):
    if not FLAGS.checkpoint_name:
        FLAGS.checkpoint_name = FLAGS.project_name
    if FLAGS.considered_phrase_type is not None and not isinstance(FLAGS.considered_phrase_type, list):
        FLAGS.considered_phrase_type = [FLAGS.considered_phrase_type]


def print_hparams(hparams, skip_patterns=None):
    """Print hparams, can skip keys based on pattern."""
    values = hparams.__dict__
    for key in sorted(values.keys()):
        if not skip_patterns or all(
                [skip_pattern not in key for skip_pattern in skip_patterns]):
            print("  %s=%s" % (key, str(values[key])))


def copy_hparams_to_outdir(hparams):
    out_dir = hparams.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ori_hparams_file = hparams.config_file
    hparams_basename = os.path.basename(ori_hparams_file)
    dst_hparams_file = os.path.join(out_dir, hparams_basename)
    shutil.copyfile(ori_hparams_file, dst_hparams_file)


def write_hparams_to_outdir(hparams):
    out_dir = hparams.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ori_hparams_file = hparams.config_file
    hparams_basename = os.path.basename(ori_hparams_file)


def load_hparams(model_dir):
    """Load hparams from an existing model directory."""
    hparams_file = os.path.join(model_dir, "hparams")
    if os.path.exists(hparams_file):
        print("# Loading hparams from %s" % hparams_file)
        with open(hparams_file, "r") as f:
            try:
                hparams_values = json.load(f)
                hparams = argparse.Namespace()
                hparams.__dict__.update(hparams_values)
            except ValueError:
                print("  can't load hparams file")
                return None
        return hparams
    else:
        return None

def standard_uniform_weights(m, std=0.1):
    for param in m.parameters():
        if param.dim() > 1:
            nn.init.uniform_(param.data, -std, std)
        else:
            nn.init.constant_(param.data, val=0)

def uniform_weights(m, std=0.1):
    for param in m.parameters():
        param.data.uniform_(-std, std)

    # for name, param in m.named_parameters():
        # nn.init.uniform_(param.data, -0.08, 0.08)
        # nn.init.uniform_(param.data, -0.1, 0.1)
        # nn.init.orthogonal_(param.data)


def xavier_uniform_weights(m):
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def kaiming_init_weights(m):
    for name, param in m.named_parameters():
        # nn.init.uniform_(param.data, -0.08, 0.08)
        # print(name, param.data.shape)
        # nn.init.kaiming_normal(param.data)
        # nn.init.orthogonal_(param.data)
        if name.endswith("weight"):
            nn.init.kaiming_normal_(param.data)
        elif name.endswith("bias"):
            nn.init.constant_(param.data, val=0)
        else:
            raise ValueError("Unknown parameters %s" % name)


# def load_model(model, path, checkpoint_name):
#     model_path = os.path.join(path, checkpoint_name)
#     if os.path.exists(model_path):
#         print("load existing model...")
#         model.load_state_dict(torch.load(model_path))
#         return True
#     else:
#         print("create new model...")
#         model.apply(init_weights)
#         return False
#
# def save_model(model, path, checkpoint_name):
#     model_path = os.path.join(path, checkpoint_name)
#     torch.save(model.state_dict(), model_path)

def load_or_create_model(model,
                         path,
                         checkpoints_filename='checkpoints',
                         initial_func_name='uniform'):
    if_loaded = load_model(model, path, checkpoints_filename)
    if if_loaded:
        return True
    else:
        if initial_func_name == 'uniform':
            print_out("create new model with uniform initializer function...")
            model.apply(uniform_weights)
        elif initial_func_name == 'standard_uniform':
            print_out("create new model with standard uniform initializer function...")
            model.apply(standard_uniform_weights)
        elif initial_func_name == 'kaiming':
            print_out("create new model with kaiming initializer function...")
            model.apply(kaiming_init_weights)
        elif initial_func_name == 'xavier_uniform':
            print_out("create new model with xavier_uniform initializer function...")
            model.apply(xavier_uniform_weights)
        elif initial_func_name == 'self_designed':
            print_out("create new model with self-designed initializer function...")
            model._reset_parameters()
        else:
            if initial_func_name.lower() == 'none':
                print_out("model created without initialization.")
            else:
                raise ValueError("no such initializer function.")

        return False


def load_model(model, path, checkpoints_filename='checkpoints'):
    if os.path.isfile(path):
        print_out("load existing model from provided checkpoint: %s" % path)
        model.load_state_dict(torch.load(path))
        return True

    ckp_record_file = os.path.join(path, checkpoints_filename)
    if os.path.exists(ckp_record_file):
        with open(ckp_record_file, 'r') as fp:
            ckps = fp.readlines()
            ckps = [c.strip() for c in ckps]
        cur_ckpt_path = os.path.join(path, ckps[0])
        print_out("load existing model from checkpoint record file: %s" % cur_ckpt_path)
        model.load_state_dict(torch.load(cur_ckpt_path))
        return True
    else:
        print_out('no existing model to load. %s' % ckp_record_file)
        return False


def load_model_state_dict(path, checkpoints_filename='checkpoints'):
    if os.path.isfile(path):
        print("load existing model from provided checkpoint: %s" % path)
        return torch.load(path)

    ckp_record_file = os.path.join(path, checkpoints_filename)
    if os.path.exists(ckp_record_file):
        with open(ckp_record_file, 'r') as fp:
            ckps = fp.readlines()
            ckps = [c.strip() for c in ckps]
        cur_ckpt_path = os.path.join(path, ckps[0])
        print("load existing model from checkpoint record file: %s" % cur_ckpt_path)
        return torch.load(cur_ckpt_path)
    else:
        return None


def save_model(model, path, checkpoint_name, global_step, keep_number=10, checkpoints_filename='checkpoints'):
    import os
    # first save the model
    saved_ckp_name = "%s.ckpt-%s.pth" % (checkpoint_name, str(global_step))
    model_path = os.path.join(path, saved_ckp_name)
    torch.save(model.state_dict(), model_path)

    # second read the save history
    ckp_record_file = os.path.join(path, checkpoints_filename)
    if os.path.exists(ckp_record_file):
        with open(ckp_record_file, 'r') as fp:
            ckps = fp.readlines()
            ckps = [c.strip() for c in ckps]
    else:
        ckps = [saved_ckp_name]

    # third add current checkpoint to the history
    ckps[0] = saved_ckp_name
    ckps.append(saved_ckp_name)

    # forth remove the overdue checkpoint history
    if len(ckps) > keep_number + 1:
        removed_ckp = os.path.join(path, ckps.pop(1))
        if os.path.exists(removed_ckp):
            os.remove(removed_ckp)

    # fifth write history records to file
    with open(ckp_record_file, 'w') as fp:
        for c in ckps:
            fp.write(c)
            fp.write('\n')


def average_lastN(running_model, model_path, N=3, checkpoints_filename='checkpoints',
                  averaged_checkpoints_filename='averaged_checkpoints'):
    # first read the save history
    ckp_record_file = os.path.join(model_path, checkpoints_filename)
    if os.path.exists(ckp_record_file):
        with open(ckp_record_file, 'r') as fp:
            ckps = fp.readlines()
            ckps.reverse()
            ckps = [c.strip() for c in ckps]
    else:
        print('No checkpoints file.')
        return
    if N is None:
        N = len(ckps) - 1
    if N > len(ckps) - 1:
        print("N=%d is bigger than the number(%d) of checkpoints." % (N, len(ckps) - 1))

    # read last N checkpoint names, and record its checkpoint numbers.
    ckpt_names = []
    ckpt_codes = []
    pattern = re.compile(r'\d+')
    for i, ckpt_name in enumerate(ckps):
        if i < N:
            print(ckpt_name)
            ckpt_names.append(ckpt_name)
            ckpt_codes.append(pattern.findall(ckpt_name)[0])
        else:
            break

    ckpt_path = [os.path.join(model_path, ckpt_name) for ckpt_name in ckpt_names]
    num_model = len(ckpt_path)
    save_name = ckpt_names[0].split('.')[0]
    global_step = '-'.join(ckpt_codes)

    # read N models
    models = []
    avg_model = load_model_state_dict(ckpt_path[0])
    for mp in ckpt_path[1:]:
        model = load_model_state_dict(mp)
        models.append(model)

    # average N models
    printname = None
    for key in avg_model:
        if key == printname:
            print(avg_model[key])
        for model in models:
            avg_model[key] += model[key]
            if key == printname:
                print(model[key])

        avg_model[key] = avg_model[key] / num_model
        if key == printname:
            print(avg_model[key])

    # save N models
    running_model.load_state_dict(avg_model)
    save_model(running_model, model_path, save_name, global_step, checkpoints_filename=averaged_checkpoints_filename)

def remove_json_notes(fp):
    import re
    dst_json_str = ''
    lines = fp.readlines()
    for l in lines:
        dst_json_str += re.sub(r'\/\*.*\*\/', "", l)
    return dst_json_str

def format_results(mode, experiment_dir, dataset_file, results):
    line_list = list()
    line_list.append(mode)
    _, exp_dest_dir = os.path.split(experiment_dir)
    line_list.append(exp_dest_dir)
    _, data_dest_dir = os.path.split(dataset_file)
    line_list.append(data_dest_dir)
    for k in results:
        v = results[k]
        resultline = "%s:%.2f" % (k, v)
        line_list.append(resultline)
    return " ".join(line_list)

def record_results(record_file, record_line, new_line=True):
    p, f = os.path.split(record_file)
    if not os.path.isdir(p):
        os.makedirs(p)
    with open(record_file, 'a') as fp:
        fp.write(record_line)
        if new_line:
            fp.write('\n')

def test():
    f = 'experiment/data/test.tok.lc.1000.en'
    dict_file = 'experiment/data/vocab.tok.lc.en'
    w2id, length = build_word2iddict(dict_file)
    s, e, l = text_preprocess(f, w2id)

    for i, (ss, ee, ll) in enumerate(zip(s, e, l)):
        print(ss)
        print(ee)
        print(ll)
        if i > 5:
            break


if __name__ == "__main__":
    test()
