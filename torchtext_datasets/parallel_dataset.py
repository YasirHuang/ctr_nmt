# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:parallel_dataset.py
@time:2020-09-23 
@desc:
'''
import os

from torchtext import data
from torchtext.data import Example
from contextlib import ExitStack


class NormalDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        if hasattr(ex, 'tgt'):
            return data.interleave_keys(len(ex.src), len(ex.tgt))
        else:
            return len(ex.src)

class ParallelDataset(NormalDataset):
    """ Define a N-parallel dataset: supports abitriry numbers of input streams"""

    def __init__(self, path, exts, fields, max_len=None, **kwargs):
        assert len(exts) == len(fields), 'N parallel dataset must match'
        self.N = len(fields)

        if not isinstance(fields[0], (tuple, list)):
            newfields = [('src', fields[0]), ('tgt', fields[1])]
            for i in range(len(exts) - 2):
                newfields.append(('extra_{}'.format(i), fields[2 + i]))
            # self.fields = newfields
            fields = newfields

        paths = tuple(os.path.expanduser(path + '.' + x) for x in exts)
        # self.max_len = max_len
        examples = []

        with ExitStack() as stack:
            files = [stack.enter_context(open(fname, encoding='utf-8')) for fname in paths]
            for i, lines in enumerate(zip(*files)):
                lines = [line.strip() for line in lines]
                if not any(line == '' for line in lines):
                    example = Example.fromlist(lines, fields)
                    # examples.append(example)
                    if max_len is None:
                        examples.append(example)
                    elif len(example.src) <= max_len and len(example.tgt) <= max_len:
                        examples.append(example)
        super(ParallelDataset, self).__init__(examples, fields, **kwargs)
