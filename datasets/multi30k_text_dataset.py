# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:mca_translator_dataset.py
@time:2020-04-19 
@desc:
'''

import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader

from datasets.utils import process_sentence, custom_collate

class Multi30KTextDataset(Dataset.Dataset):
    def __init__(self,
                 src_sentences,
                 tgt_sentences,
                 src_word_to_id_dict,
                 tgt_word_to_id_dict,
                 src_sos='<sos>',
                 tgt_sos='<sos>',
                 eos='<eos>',
                 unk='<unk>'
                 ):
        '''

        :param src_sentences:
        :param tgt_sentences:
        :param src_word_to_id_dict:
        :param tgt_word_to_id_dict:
        :param sos:
        :param eos:
        :param unk:
        '''
        assert len(src_sentences) == len(tgt_sentences), \
            'src_sentences=%d, tgt_sentences=%d' % (len(src_sentences), len(tgt_sentences))

        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_word_to_id_dict = src_word_to_id_dict
        self.tgt_word_to_id_dict = tgt_word_to_id_dict

        self.src_sos = src_sos
        self.tgt_sos = tgt_sos
        self.eos = eos
        self.unk = unk

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        # prepare src sentence
        sos_src, eos_src = process_sentence(self.src_sentences[index],
                                            self.src_word_to_id_dict,
                                            self.src_sos, self.eos, self.unk)
        # prepare tgt sentence
        sos_tgt, eos_tgt = process_sentence(self.tgt_sentences[index],
                                            self.tgt_word_to_id_dict,
                                            self.tgt_sos, self.eos, self.unk)
        src_sent_len = torch.Tensor([sos_src.shape[0]]).int()
        tgt_sent_len = torch.Tensor([sos_tgt.shape[0]]).int()

        return sos_src, eos_src, sos_tgt, eos_tgt, src_sent_len, tgt_sent_len


def filter(src, tgt, min_sentence_length, max_sentence_length):
    newsrc = []
    newtgt = []
    for s, t in zip(src, tgt):
        slen = len(s.strip().split())
        tlen = len(t.strip().split())
        if slen >= min_sentence_length and slen <= max_sentence_length and tlen >= min_sentence_length and tlen <= max_sentence_length:
            newsrc.append(s)
            newtgt.append(t)
    return newsrc, newtgt


def build_dataloader(src_sentences_file,
                     tgt_sentences_file,
                     src_word_to_id_dict,
                     tgt_word_to_id_dict,
                     batch_size,
                     min_sentence_length=0,
                     max_sentence_length=50,
                     shuffle=True,
                     num_workers=1,
                     src_sos='<sos>',
                     tgt_sos='<sos>',
                     eos='<eos>',
                     unk='<unk>'):
    '''

    :param src_sentences_file:
    :param tgt_sentences_file:
    :param src_word_to_id_dict:
    :param tgt_word_to_id_dict:
    :param batch_size:
    :param min_sentence_length:
    :param max_sentence_length:
    :param shuffle:
    :param num_workers:
    :param sos:
    :param eos:
    :param unk:
    :return:
    '''

    with open(src_sentences_file, 'r') as fp:
        src_sentences = fp.readlines()
    with open(tgt_sentences_file, 'r') as fp:
        tgt_sentences = fp.readlines()

    src_sentences, tgt_sentences = filter(src_sentences,
                                          tgt_sentences,
                                          min_sentence_length,
                                          max_sentence_length)

    dataset = Multi30KTextDataset(src_sentences,
                                  tgt_sentences,
                                  src_word_to_id_dict,
                                  tgt_word_to_id_dict,
                                  src_sos, tgt_sos, eos, unk)

    dataloader = DataLoader.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       collate_fn=custom_collate)
    return dataloader
