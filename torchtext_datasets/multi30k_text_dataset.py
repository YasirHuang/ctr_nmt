# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:mca_translator_dataset.py
@time:2020-04-19 
@desc:
'''
import os

from torchtext import data
from torchtext.data import Example
from torchtext_datasets.fields import NormalField

from contextlib import ExitStack
from utils.config import DATA_PART


class Multi30KTextDatasetProfile:
    def __init__(self,
                 src,
                 tgt,
                 train_prefix,
                 val_prefix,
                 test_prefix,
                 batch_size,
                 batch_first=True,
                 include_lengths=True,
                 share_vocab=False,
                 max_sentence_length=50,
                 src_sos='<sos>',
                 tgt_sos='<sos>',
                 eos='<eos>',
                 pad='<pad>',
                 unk='<unk>'
                 ):
        self.src = src
        self.tgt = tgt
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.test_prefix = test_prefix
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.include_lengths = include_lengths
        self.share_vocab = share_vocab
        self.max_sentence_length = max_sentence_length
        self.src_sos = src_sos
        self.tgt_sos = tgt_sos
        self.eos = eos
        self.pad = pad
        self.unk = unk

    def build_fields(self):
        id_field = data.Field(sequential=False, use_vocab=False)
        src_field = NormalField(init_token=self.src_sos,
                                eos_token=self.eos,
                                unk_token=self.unk,
                                pad_token=self.pad,
                                batch_first=self.batch_first,
                                include_lengths=self.include_lengths)
        tgt_field = src_field \
            if self.share_vocab and \
               self.src_sos == self.tgt_sos \
            else NormalField(init_token=self.tgt_sos,
                             eos_token=self.eos,
                             unk_token=self.unk,
                             pad_token=self.pad,
                             batch_first=self.batch_first,
                             include_lengths=self.include_lengths)
        fields = [("id", id_field), ("src", src_field), ("tgt", tgt_field)]
        suffixes = (self.src, self.tgt)
        setattr(self, "fields", fields)
        setattr(self, "suffixes", suffixes)
        return fields, suffixes

    def get_prefix(self, part):
        if part == DATA_PART.Train:
            return self.train_prefix
        elif part == DATA_PART.Val:
            return self.val_prefix
        elif part == DATA_PART.Test:
            return self.test_prefix
        else:
            raise ValueError("Unknown dataset part %s" % part)

# TODO: for test mode, there should be no limitation of sentence length.
    def load_examples(self, part):
        suffixes = getattr(self, "suffixes")
        fields = getattr(self, "fields")

        assert isinstance(fields[0], (tuple, list)), \
            'field in fields must be a tuple/list , coupled with field name'

        paths = tuple(os.path.expanduser(self.get_prefix(part) + '.' + x) for x in suffixes)
        # self.max_len = max_len
        examples = []

        with ExitStack() as stack:
            files = [stack.enter_context(open(fname, encoding='utf-8')) for fname in paths]
            for i, lines in enumerate(zip(*files)):
                lines = [line.strip() for line in lines]
                lines.insert(0, i)
                if not any(line == '' for line in lines):
                    example = Example.fromlist(lines, fields)
                    # examples.append(example)
                    max_len = self.max_sentence_length
                    if max_len is None:
                        examples.append(example)
                    elif len(example.src) <= max_len and len(example.tgt) <= max_len:
                        examples.append(example)
        return examples

    def pack_data(self, data, part):
        id = data.id
        src, src_len = data.src
        tgt, tgt_len = data.tgt

        if self.batch_first:
            sos_src = src[:,:-1]
            eos_src = src[:,1:]
            sos_tgt = tgt[:,:-1]
            eos_tgt = tgt[:,1:]
        else:
            sos_src = src[:-1]
            eos_src = src[1:]
            sos_tgt = tgt[:-1]
            eos_tgt = tgt[1:]
        src_len = src_len.unsqueeze(1) - 1
        tgt_len = tgt_len.unsqueeze(1) - 1

        return [sos_src, eos_src, sos_tgt, eos_tgt, src_len, tgt_len]

        # sos_src = packed_data[0].to(device)
        # eos_src = packed_data[1].to(device)
        # sos_tgt = packed_data[2].to(device)
        # eos_tgt = packed_data[3].to(device)
        # src_len = packed_data[4].squeeze(1).to(device)
        # tgt_len = packed_data[5].squeeze(1).to(device)
