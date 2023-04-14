# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:fields.py
@time:2020-09-19 
@desc:
'''
import torch

from torchtext import data, vocab
from collections import OrderedDict, Counter
from itertools import chain

# load the dataset + reversible tokenization
class NormalField(data.Field):
    # def __init__(self, sequential=True, use_vocab=True, src_init_token=None, tgt_init_token=None,
    #          eos_token=None, fix_length=None, dtype=torch.long,
    #          preprocessing=None, postprocessing=None, lower=False,
    #          tokenize=(lambda s: s.split()), include_lengths=False,
    #          batch_first=False, pad_token="<pad>", unk_token="<unk>",
    #          pad_first=False, truncate_first=False, stop_words=None,
    #          is_target=False):
    #
    #     self.src_init_token = src_init_token
    #     self.tgt_init_token = tgt_init_token
    #     super(NormalField, self).__init__(sequential=sequential, use_vocab=use_vocab, init_token=src_init_token,
    #          eos_token=eos_token, fix_length=fix_length, dtype=dtype,
    #          preprocessing=preprocessing, postprocessing=postprocessing, lower=lower,
    #          tokenize=tokenize, include_lengths=include_lengths,
    #          batch_first=batch_first, pad_token=pad_token, unk_token=unk_token,
    #          pad_first=pad_first, truncate_first=truncate_first, stop_words=stop_words,
    #          is_target=is_target)

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        if isinstance(args[0], vocab.Vocab):
            self.vocab = args[0]
            return
        if isinstance(args[0], Counter):
            counter = args[0]
        else:
            counter = Counter()
            sources = []
            for arg in args:
                sources += [getattr(arg, name) for name, field in
                                arg.fields.items() if field is self]
            for data in sources:
                for x in data:
                    if not self.sequential:
                        x = [x]
                    try:
                        counter.update(x)
                    except TypeError:
                        counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token] + kwargs.pop('specials', [])
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def reverse(self, batch, unbpe=True, returen_token=False):
        if not self.batch_first:
            batch.t_()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch] # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch] # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        if unbpe:
            batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
        else:
            batch = [" ".join(filter(filter_special, ex)) for ex in batch]

        if returen_token:
            batch = [ex.split() for ex in batch]
        return batch
