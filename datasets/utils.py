# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:utils.py
@time:2020-01-07 
@desc:
'''
import numpy as np
import torch
import re

from torch._six import container_abcs, string_classes, int_classes

def word2id(word, word_to_id_dict, unk='<unk>'):
    if word.strip() in word_to_id_dict:
        return word_to_id_dict[word]
    else:
        return word_to_id_dict[unk]


def concat_frime(kaldi_sentences, audio_cat_size):
    ret = []
    for sent in kaldi_sentences:
        sent_frime_len = sent.shape[0]
        pad_len = (sent_frime_len // audio_cat_size + 1) * audio_cat_size - sent_frime_len
        padding = np.zeros([pad_len, sent.shape[-1]])
        s = np.concatenate([sent, padding], axis=0)
        ret.extend(np.reshape(s, [len(s) // audio_cat_size, audio_cat_size * s.shape[-1]]))
    return np.array(ret)


def process_sentence(sentence, word_to_id_dict, sos, eos, unk):
    sent = [word2id(w, word_to_id_dict, unk) for w in sentence.strip().split()]
    sos_sent = [word2id(sos, word_to_id_dict, unk)]
    sos_sent.extend(sent)
    eos_sent = sent
    eos_sent.append(word2id(eos, word_to_id_dict, unk))
    return torch.Tensor(sos_sent).long(), torch.Tensor(eos_sent).long()


def process_sentence_with_se(sentence, word_to_id_dict, sos, eos, unk):
    sent = [word2id(w, word_to_id_dict, unk) for w in sentence.strip().split()]
    sos_sent = [word2id(sos, word_to_id_dict, unk)]
    sos_sent.extend(sent)
    ful_sent = sos_sent
    ful_sent.append(word2id(eos, word_to_id_dict, unk))
    return torch.Tensor(ful_sent[:-1]).long(), torch.Tensor(ful_sent).long()


# the same as conject_postag in torchtext_datasets.utils.conject_postag
def conject_postag(src_sent, sent_doc):
    word_idx = 0
    conject = False
    ret_pos = []
    for token in sent_doc:
        word_from_doc = token.text
        word_from_src = src_sent[word_idx:word_idx + len(word_from_doc)]
        if word_from_doc != word_from_src:
            raise ValueError("wrong alignment! src={}, doc={}".format(word_from_src, word_from_doc))

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



def padding(data, padding_value=None):
    if not padding_value:
        try:
            padding_value = torch.zeros_like(data[0][0])
        except:
            print(data)
            #print(np.shape(data))
            for d in data:
                if torch.is_tensor(d):
                    print(d.shape)

            def printdtype(data, layer):
                if isinstance(data, (tuple, list)):
                    print(layer, np.shape(data), type(data))
                    for d in data:
                        printdtype(d, layer + 1)
                else:
                    print(layer, np.shape(data), type(data))

            printdtype(data[1:], 0)
    max_seq_len = 0
    for d in data:
        if d.shape[0] > max_seq_len:
            max_seq_len = d.shape[0]
    ret = []
    for d in data:
        pad_len = max_seq_len - d.shape[0]
        if pad_len > 0:
            r = torch.cat([d, torch.repeat_interleave(padding_value.unsqueeze(0), pad_len, dim=0)])
        else:
            r = d
        ret.append(r)
    return tuple(ret)

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def custom_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    #     print('elem_type: ', elem_type)
    #     print('elem len : ', len(elem))
    #     print('elem[0]shape: ', elem[0].shape)
    #     print('batch len', len(batch))

    if isinstance(elem, torch.Tensor):
        #         print('dataloader')
        #         print('batch_type: ', type(batch))
        #         print('batch_len : ', len(batch))
        #         print(batch)
        batch = padding(batch)
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
