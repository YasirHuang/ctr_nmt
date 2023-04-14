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
import numpy as np

from datasets.utils import process_sentence, custom_collate

class Multi30KDataset(Dataset.Dataset):
    def __init__(self,
                 src_sentences,
                 tgt_sentences,
                 global_features,
                 local_features,
                 src_word_to_id_dict,
                 tgt_word_to_id_dict,
                 src_sos='<sos>',
                 tgt_sos='<sos>',
                 eos='<eos>',
                 unk='<unk>',
                 load_global_features=True,
                 load_local_features=True,
                 ):
        '''

        :param src_sentences:
        :param tgt_sentences:
        :param global_features:
        :param local_features:
        :param src_word_to_id_dict:
        :param tgt_word_to_id_dict:
        :param sos:
        :param eos:
        :param unk:
        '''
        assert len(src_sentences) == len(tgt_sentences), \
            'src_sentences=%d, tgt_sentences=%d' % (len(src_sentences), len(tgt_sentences))
        # assert len(src_sentences) <= len(global_features), \
        #     'src_sentences=%d, global_features=%d' % (len(src_sentences), len(global_features))
        # assert len(src_sentences) <= len(local_features), \
        #     'src_sentences=%d, local_features=%d' % (len(src_sentences), len(local_features))
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.global_features = global_features
        self.local_features = local_features
        self.src_word_to_id_dict = src_word_to_id_dict
        self.tgt_word_to_id_dict = tgt_word_to_id_dict

        self.src_sos = src_sos
        self.tgt_sos = tgt_sos
        self.eos = eos
        self.unk = unk

        self.load_local_features = load_local_features
        self.load_global_features = load_global_features

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
        if self.load_global_features:
            global_feature = torch.Tensor(self.global_features[index])
        if self.load_local_features:
            local_feature = torch.Tensor(self.local_features[index])
            local_feat_len = torch.Tensor([local_feature.shape[0]]).int()
        src_sent_len = torch.Tensor([sos_src.shape[0]]).int()
        tgt_sent_len = torch.Tensor([sos_tgt.shape[0]]).int()

        if self.load_global_features and not self.load_local_features:
            return sos_src, eos_src, \
               sos_tgt, eos_tgt, \
               src_sent_len, tgt_sent_len, \
               global_feature
        elif self.load_local_features and not self.load_global_features:
            return sos_src, eos_src, \
               sos_tgt, eos_tgt, \
               src_sent_len, tgt_sent_len, \
               local_feature, local_feat_len
        elif self.load_global_features and self.load_local_features:
            return sos_src, eos_src, \
               sos_tgt, eos_tgt, \
               src_sent_len, tgt_sent_len, \
               global_feature, \
               local_feature, local_feat_len
        else:
            return sos_src, eos_src, \
               sos_tgt, eos_tgt, \
               src_sent_len, tgt_sent_len


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
                     global_image_feature_file,
                     local_image_feature_file,
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
                     unk='<unk>',
                     load_global_features=True,
                     load_local_features=True):
    '''

    :param src_sentences_file:
    :param tgt_sentences_file:
    :param global_image_feature_file:
    :param local_image_feature_file:
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

    if load_global_features:
        global_image_features = np.load(global_image_feature_file)
    else:
        global_image_features = None
    if load_local_features:
        local_image_features = np.load(local_image_feature_file)
    else:
        local_image_features = None

    with open(src_sentences_file, 'r') as fp:
        src_sentences = fp.readlines()
    with open(tgt_sentences_file, 'r') as fp:
        tgt_sentences = fp.readlines()

    src_sentences, tgt_sentences = filter(src_sentences,
                                          tgt_sentences,
                                          min_sentence_length,
                                          max_sentence_length)

    dataset = Multi30KDataset(src_sentences,
                              tgt_sentences,
                              global_image_features,
                              local_image_features,
                              src_word_to_id_dict,
                              tgt_word_to_id_dict,
                              src_sos, tgt_sos, eos, unk,
                              load_global_features=load_global_features,
                              load_local_features=load_local_features)

    dataloader = DataLoader.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       collate_fn=custom_collate)
    return dataloader
