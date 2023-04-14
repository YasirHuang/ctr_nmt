# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:utils.py
@time:2020-09-25 
@desc:
'''
from torchtext import data


def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(len(new.src), len(new.tgt), prev_max_len) * i


def dyn_batch_without_padding(new, i, sofar):
    return sofar + max(len(new.src), len(new.trg))


def sort_key(ex):
    return data.interleave_keys(len(ex.src), len(ex.tgt))


# the same as conject_postag in datasets.utils.conject_postag
def conject_postag(src_sent, sent_doc):
    word_idx = 0
    conject = False
    ret_pos = []
    for token in sent_doc:
        word_from_doc = token.text
        word_from_src = src_sent[word_idx:word_idx + len(word_from_doc)]
        if word_from_doc != word_from_src:
            raise ValueError("wrong alignment!")

        if conject:
            ret_pos[-1] = '-'.join([ret_pos[-1], token.pos_])
        else:
            ret_pos.append(token.pos_)
        if word_idx + len(word_from_doc) < len(src_sent):
            if src_sent[word_idx + len(word_from_doc)] == ' ':
                word_idx += len(word_from_doc) + 1
                conject = False
            else:
                word_idx += len(word_from_doc)
                conject = True
    if not len(ret_pos) == len(src_sent.strip().split()):
        raise ValueError('wrong conject!')
    return ret_pos
