# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:average_models.py
@time:2021/12/7 
@desc:
'''
from __future__ import division, absolute_import, print_function
import os
import re
import six
import torch
from collections import OrderedDict
from copy import deepcopy
from utils.checkpoint import Checkpoint, load_model_state_dict, save_model_state_dict

def average_lastN(path,
                  N=3,
                  checkpoints_filename='checkpoints',
                  averaged_checkpoint_names='avg_checkpoint',
                  averaged_checkpoints_filename='averaged_checkpoints'):
    if path is None:
        path = os.environ.get('OUT_DIR')
    ckp_record_file = os.path.join(path, checkpoints_filename)
    checkpoint_recorder = Checkpoint(ckp_record_file)

    if N is None:
        N = len(checkpoint_recorder)
    if N > len(checkpoint_recorder):
        N = len(checkpoint_recorder)
        print("N=%d is bigger than the number(%d) of checkpoints." % (N, len(checkpoint_recorder)))

    # read last N checkpoint names, and record its checkpoint numbers.
    ckpt_names = []
    ckpt_codes = []
    # FIXME:
    # ckpt_names = [checkpoint_recorder[-i] for i in range(1, N+1)]
    pattern = re.compile(r'\d+')
    for i in range(N):
        ckpt_name = checkpoint_recorder.pop()
        ckpt_names.append(ckpt_name)
        ckpt_codes.append(pattern.findall(ckpt_name)[0])

    ckpt_path = [os.path.join(path, ckpt_name) for ckpt_name in ckpt_names]
    global_step = '-'.join(ckpt_codes)

    # read N models
    ckpts = []
    for mp in ckpt_path:
        ckpt = load_model_state_dict(mp)
        ckpts.append(ckpt)

    # average N models
    def search_and_average(*args):
        if isinstance(args[0], (dict, OrderedDict)):
            ret = type(args[0])()
            for k in six.iterkeys(args[0]):
                values = [arg[k] for arg in args]
                ret[k] = search_and_average(*values)
            return ret
        if isinstance(args[0], torch.Tensor):
            n = len(args)
            ret = args[0] / n
            for arg in args[1:]:
                ret += arg / n
            return ret
        else:
            return args[0]

    averaged = search_and_average(*ckpts)
    save_model_state_dict(averaged,
                          path=path,
                          checkpoint_name=averaged_checkpoint_names,
                          global_step=global_step,
                          keep_number=1,
                          checkpoints_filename=averaged_checkpoints_filename)


    # printname = None
    # for key in six.iterkeys(avg_model):
    #     if key == printname:
    #         print(avg_model[key])
    #     for model in models:
    #         avg_model[key] += model[key]
    #         if key == printname:
    #             print(model[key])
    #
    #     avg_model[key] = avg_model[key] / N
    #     if key == printname:
    #         print(avg_model[key])
    #
    # # save N models
    # running_model.load_state_dict(avg_model)
    # save_model(running_model, model_path, save_name, global_step, checkpoints_filename=averaged_checkpoints_filename)
