# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:multi30k_boundingbox_dataset.py
@time:2020-08-10 
@desc:
'''

try:
    import spacy
except Exception:
    pass
import json

import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np
import os

from datasets.utils import process_sentence, conject_postag, custom_collate


class Multi30KBoundingBoxDataset(Dataset.Dataset):
    def __init__(self,
                 splits,
                 src_sentences_json,
                 tgt_sentences,
                 src_word_to_id_dict,
                 tgt_word_to_id_dict,
                 image_features,
                 key_word_pos_list,
                 key_word_list,
                 spacy_nlp=None,
                 real_time_parse=False,
                 considered_phrase_type=None,
                 key_word_own_image=False,
                 average_multiple_images=False,
                 src_sos='<sos>',
                 tgt_sos='<sos>',
                 eos='<eos>',
                 unk='<unk>'
                 ):
        '''

        :param splits:
        :param src_sentences_json:
        :param tgt_sentences:
        :param src_word_to_id_dict:
        :param tgt_word_to_id_dict:
        :param image_features:
        :param key_word_pos_list:
        :param key_word_list:
        :param spacy_nlp:
        :param real_time_parse:
        :param considered_phrase_type: set to None to allow all type of phrases
        :param key_word_own_image:
        :param average_multiple_images:
        :param src_sos:
        :param tgt_sos:
        :param eos:
        :param unk:
        '''
        assert len(src_sentences_json) == len(tgt_sentences), \
            'src_sentences=%d, tgt_sentences=%d' % (len(src_sentences_json), len(tgt_sentences))

        self.splits = splits
        self.src_sentences_json = src_sentences_json
        self.tgt_sentences = tgt_sentences
        self.src_word_to_id_dict = src_word_to_id_dict
        self.tgt_word_to_id_dict = tgt_word_to_id_dict
        self.image_features = image_features
        self.nlp = spacy_nlp
        self.real_time_parse = real_time_parse
        self.considered_phrase_type = considered_phrase_type

        self.key_word_own_image = key_word_own_image
        self.key_word_pos_list = key_word_pos_list
        self.key_word_list = key_word_list
        self.average_multiple_images = average_multiple_images

        self.src_sos = src_sos
        self.tgt_sos = tgt_sos
        self.eos = eos
        self.unk = unk

    def __len__(self):
        return len(self.splits)

    def __tag_source_sentence__(self, src_sentence):
        phrases = sorted(src_sentence['phrases'], key=lambda item: item['first_word_index'])
        nlp = self.nlp
        sentence = src_sentence['sentence'].strip().split()
        if self.key_word_own_image:
            sentence_doc = nlp(src_sentence['sentence'])
            sentence_pos = conject_postag(src_sentence['sentence'], sentence_doc)
            assert len(sentence) == len(sentence_pos)
        tag = [None] * len(sentence)
        for phrase in phrases:
            first_word_index = phrase['first_word_index']
            phrase_len = len(phrase['phrase'].strip().split())
            for i in range(phrase_len):
                if self.key_word_own_image:
                    if sentence_pos[i + first_word_index] in self.key_word_pos_list or \
                            (self.key_word_list is not None and sentence[i + first_word_index] in self.key_word_list):
                        tag[i + first_word_index] = phrase['phrase_id']
                else:
                    tag[i + first_word_index] = phrase['phrase_id']
        return tag

    def __getitem__(self, index):
        split_id = self.splits[index]
        src_sentence_json = self.src_sentences_json[split_id]
        phrase_dict = {}
        for p in src_sentence_json["phrases"]:
            phrase_dict[p["phrase_id"]] = {"first_word_index": p["first_word_index"],
                                           "phrase_type": p["phrase_type"],
                                           "phrase": p["phrase"]}
        # get word tag, each word position will be marked as None or "phrase_id"
        if self.real_time_parse:
            word_tag = self.__tag_source_sentence__(src_sentence_json)
        elif self.key_word_own_image and 'key_tag' in src_sentence_json:
            word_tag = src_sentence_json['key_tag']
        elif not self.key_word_own_image and 'all_tag' in src_sentence_json:
            word_tag = src_sentence_json['all_tag']
        else:
            raise ValueError("Please provide phrase tag.")

        src_sentence = src_sentence_json['sentence'].lower()
        tgt_sentence = self.tgt_sentences[index]
        # prepare src sentences
        sos_src, eos_src = process_sentence(src_sentence,
                                            self.src_word_to_id_dict,
                                            self.src_sos, self.eos, self.unk)
        # prepare tgt sentences
        sos_tgt, eos_tgt = process_sentence(tgt_sentence,
                                            self.tgt_word_to_id_dict,
                                            self.tgt_sos, self.eos, self.unk)
        src_sent_len = torch.Tensor([sos_src.shape[0]]).int()
        tgt_sent_len = torch.Tensor([sos_tgt.shape[0]]).int()

        # prepare img contexts
        image_feature_dim = self.image_features[0][list(self.image_features[0].keys())[0]][0].shape[-1]
        feat_list = torch.zeros([src_sent_len.item(), image_feature_dim])
        feat_mask = torch.zeros([src_sent_len.item()])
        for i, phrase_id in enumerate(word_tag):
            if phrase_id is not None:
                # note: phrase_type is a list
                phrase_type = phrase_dict[phrase_id]["phrase_type"]
                if self.considered_phrase_type is not None:
                    if len(set(phrase_type) & set(self.considered_phrase_type)) == 0:
                        continue

                # the phrase_id index in image_features is save as type bytes
                # so the transformation is needed
                phrase_id = bytes(phrase_id, encoding='utf-8')
                if phrase_id in self.image_features[index]:
                    if self.average_multiple_images:
                        feat = np.average(self.image_features[index][phrase_id], axis=0)
                    else:
                        feat = self.image_features[index][phrase_id][0]
                    feat_list[i + 1] = torch.from_numpy(feat)
                    feat_mask[i + 1] = 1.0

        return sos_src, eos_src, \
               sos_tgt, eos_tgt, \
               src_sent_len, tgt_sent_len, \
               feat_list, feat_mask


def tag_source_sentence(nlp, src_sentence, key_word_own_image, key_word_list, key_word_pos_list):
    phrases = sorted(src_sentence['phrases'], key=lambda item: item['first_word_index'])
    sentence = src_sentence['sentence'].strip().split()
    if key_word_own_image:
        sentence_doc = nlp(src_sentence['sentence'])
        sentence_pos = conject_postag(src_sentence['sentence'], sentence_doc)
        assert len(sentence) == len(sentence_pos)
    tag = [None] * len(sentence)
    for phrase in phrases:
        first_word_index = phrase['first_word_index']
        phrase_len = len(phrase['phrase'].strip().split())
        for i in range(phrase_len):
            if key_word_own_image:
                if sentence_pos[i + first_word_index] in key_word_pos_list or \
                        (key_word_list is not None and sentence[i + first_word_index] in key_word_list):
                    tag[i + first_word_index] = phrase['phrase_id']
            else:
                tag[i + first_word_index] = phrase['phrase_id']
    return tag


def tag_corpus(splits, sentences_json, key_word_own_image, key_word_list, key_word_pos_list):
    tags = {}
    nlp = spacy.load('en_core_web_sm')
    for split_id in splits:
        src_sentence_json = sentences_json[split_id]
        word_tag = tag_source_sentence(nlp, src_sentence_json, key_word_own_image, key_word_list, key_word_pos_list)
        tags[split_id] = word_tag
    return tags


def build_dataloader(splits_file,
                     src_sentences_json_file,
                     tgt_sentences_file,
                     src_word_to_id_dict,
                     tgt_word_to_id_dict,
                     image_features_file,
                     key_word_pos_list_file,
                     key_word_list_file,
                     batch_size,
                     real_time_parse=False,
                     considered_phrase_type=None,
                     key_word_own_image=False,
                     average_multiple_images=False,
                     shuffle=True,
                     num_workers=1,
                     src_sos=None,
                     tgt_sos=None,
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
    with open(splits_file, 'r') as fp:
        splits = [s.strip().split('.')[0] for s in fp.readlines()]
    with open(src_sentences_json_file, 'r', encoding='utf-8') as fp:
        src_sentences_json = json.load(fp)
    with open(tgt_sentences_file, 'r') as fp:
        tgt_sentences = fp.readlines()
    with open(key_word_pos_list_file, 'r') as fp:
        key_word_pos_list = [l.strip() for l in fp.readlines()]
    if key_word_list_file is not None:
        with open(key_word_list_file, 'r') as fp:
            key_word_list = [l.strip() for l in fp.readlines()]
    else:
        key_word_list = None
    if considered_phrase_type is not None:
        if isinstance(considered_phrase_type, str):
            assert os.path.exists(considered_phrase_type), \
                'considered_phrase_type should be a file path when it is a string'
            with open(considered_phrase_type, 'r') as fp:
                cpt = [l.strip() for l in fp.readlines()]
            considered_phrase_type = cpt
        else:
            assert isinstance(considered_phrase_type, list), \
                "considered_phrase_type should be a list of phrase type"


    # sentence_preprocessed_tags = tag_corpus(splits, src_sentences_json, key_word_own_image, key_word_list, key_word_pos_list)
    nlp = spacy.load('en_core_web_sm') if real_time_parse else None

    image_features = np.load(image_features_file, allow_pickle=True, encoding='bytes')
    dataset = Multi30KBoundingBoxDataset(
        splits,
        src_sentences_json,
        tgt_sentences,
        src_word_to_id_dict,
        tgt_word_to_id_dict,
        image_features,
        key_word_pos_list,
        key_word_list,
        spacy_nlp=nlp,
        real_time_parse=real_time_parse,
        considered_phrase_type=considered_phrase_type,
        key_word_own_image=key_word_own_image,
        average_multiple_images=average_multiple_images,
        src_sos=src_sos, tgt_sos=tgt_sos, eos=eos, unk=unk)

    dataloader = DataLoader.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       collate_fn=custom_collate)
    return dataloader
