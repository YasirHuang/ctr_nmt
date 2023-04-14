# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:checkpoint.py
@time:2021/11/20 
@desc:
'''

from __future__ import division, absolute_import, print_function
import os
import torch
import sys

from utils.utils import print_out

class Checkpoint(object):

    def __init__(self, file_path, keep_number=10):
        assert keep_number > 0
        self._checkpoint_path = file_path
        self._keep_number = keep_number
        self._checkpoints = None
        self._current_checkpoint = None


        self.load_checkpoint_records()

    @property
    def current_checkpoint(self):
        return self._current_checkpoint

    @property
    def the_latest(self):
        return self._current_checkpoint

    def __len__(self):
        return len(self._checkpoints)

    def __getitem__(self, index):
        return self._checkpoints[index]

    def __iter__(self):
        return iter(self._checkpoints)


    def load_checkpoint_records(self):
        if os.path.exists(self._checkpoint_path):
            with open(self._checkpoint_path, 'r') as fp:
                ckps = fp.readlines()
                self._current_checkpoint = ckps[0].strip()
                self._checkpoints = [c.strip() for c in ckps[1:]]
        else:
            self._checkpoints = list()

    def save_checkpoint_records(self):
        with open(self._checkpoint_path, 'w') as fp:
            fp.write("{ckpt}".format(ckpt=self._current_checkpoint))
            for ckpt in self._checkpoints:
                fp.write("\n{ckpt}".format(ckpt=ckpt))

    def push(self, checkpoint):
        self._checkpoints.append(checkpoint)
        while len(self._checkpoints) > self._keep_number:
            discarded = self._checkpoints.pop(0)
            dir = os.path.dirname(self._checkpoint_path)
            discarded_ckpt_path = os.path.join(dir, discarded)
            if os.path.exists(discarded_ckpt_path):
                os.remove(discarded_ckpt_path)

        self._current_checkpoint = checkpoint

    def pop(self):
        ckpt = self._checkpoints.pop()
        if len(self._checkpoints) > 0:
            self._current_checkpoint = self._checkpoints[-1]
        else:
            self._current_checkpoint = None
        return ckpt



def load_model(model, path=None, checkpoints_filename='checkpoints'):
    if path is None:
        path = os.environ.get('OUT_DIR')
    if os.path.isfile(path):
        print_out("load existing model from provided checkpoint: %s" % path)
        model.load_state_dict(torch.load(path))
        return True

    ckp_record_file = os.path.join(path, checkpoints_filename)
    checkpoint_recorder = Checkpoint(ckp_record_file)
    if checkpoint_recorder.the_latest is not None:
        cur_ckpt_path = os.path.join(path, checkpoint_recorder.the_latest)
        print_out("load existing model from checkpoint record file: %s" % cur_ckpt_path)
        model.load_state_dict(torch.load(cur_ckpt_path))
        return True
    else:
        print_out('no existing model to load. %s' % ckp_record_file)
        return False


def load_model_state_dict(path=None, checkpoints_filename='checkpoints', map_location=None):
    if path is None:
        path = os.environ.get('OUT_DIR')
    if os.path.isfile(path):
        return torch.load(path, map_location=map_location)

    ckp_record_file = os.path.join(path, checkpoints_filename)
    checkpoint_recorder = Checkpoint(ckp_record_file)
    if checkpoint_recorder.the_latest is not None:
        cur_ckpt_path = os.path.join(path, checkpoint_recorder.the_latest)
        print_out("load existing model from checkpoint record file: %s" % cur_ckpt_path)
        return torch.load(cur_ckpt_path, map_location=map_location)
    else:
        return None


def save_model(model,
               path=None,
               checkpoint_name="checkpoints",
               global_step=0,
               keep_number=10,
               checkpoints_filename='checkpoints'):
    if path is None:
        path = os.environ.get('OUT_DIR')
    # first save the model
    saved_ckp_name = "%s.ckpt-%s.pth" % (checkpoint_name, str(global_step))
    model_path = os.path.join(path, saved_ckp_name)
    torch.save(model.state_dict(), model_path)

    ckp_record_file = os.path.join(path, checkpoints_filename)
    checkpoint_recorder = Checkpoint(ckp_record_file, keep_number)
    checkpoint_recorder.push(saved_ckp_name)
    checkpoint_recorder.save_checkpoint_records()


def save_model_state_dict(state_dict,
                          path=None,
                          checkpoint_name="checkpoints",
                          global_step=0,
                          keep_number=10,
                          checkpoints_filename='checkpoints'):
    if path is None:
        path = os.environ.get('OUT_DIR')
    # first save the model
    saved_ckp_name = "%s.ckpt-%s.pth" % (checkpoint_name, str(global_step))
    model_path = os.path.join(path, saved_ckp_name)
    torch.save(state_dict, model_path)

    ckp_record_file = os.path.join(path, checkpoints_filename)
    checkpoint_recorder = Checkpoint(ckp_record_file, keep_number)
    checkpoint_recorder.push(saved_ckp_name)
    checkpoint_recorder.save_checkpoint_records()