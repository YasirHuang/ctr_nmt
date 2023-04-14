# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:multi30k_boundingbox_dataset.py
@time:2020-09-25 
@desc:
'''
import os
import json
import numpy as np

try:
    import spacy
except Exception:
    pass

import torch
from torchtext import data
from torchtext_datasets.fields import NormalField
from torchtext_datasets.utils import conject_postag


from utils.config import DATA_PART


def tok(s):
    if isinstance(s, (list, tuple)):
        return s

# TODO: This redundant Field should be deleted
class BBXField(data.Field):
    def build_vocab(self, *args, **kwargs):
        assert "boundingbox_features" in kwargs
        assert "average_multiple_images" in kwargs
        bbx = kwargs['boundingbox_features']
        if_avg = kwargs['average_multiple_images']
        pad_token = self.pad_token if self.pad_token is not None else ""
        pad_vec = np.zeros_like(bbx[0][list(bbx[0].keys())[0]])
        setattr(self, "pad_vector", pad_vec)
        feat_dict = {pad_token: pad_vec}
        for sentence_features in bbx:
            for k in sentence_features.keys():
                encoded_k = str(k, encoding='utf-8')
                if encoded_k in feat_dict:
                    raise ValueError("Key already exists % ." % k)
                feat_dict[encoded_k] = np.average(sentence_features[k], axis=0) if if_avg else sentence_features[k][0]

        setattr(self, "boundingbox_feature_dict", feat_dict)

    def numericalize(self, arr, device=None):
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        # print(self.boundingbox_feature_dict.keys())

        feat_dim = self.boundingbox_feature_dict[self.pad_token].shape[-1]
        var = torch.zeros([np.shape(arr)[0], np.shape(arr)[1], feat_dim])
        if self.use_vocab and self.sequential:
            if self.sequential:
                # var = []
                for i, ex in enumerate(arr):
                    print("example is:", ex)
                    a = []
                    for j, x in enumerate(ex):
                        x = self.pad_token if x is None else x
                        x = x if x in self.boundingbox_feature_dict.keys() else self.pad_token
                        # x = bytes(self.pad_token if x is None else x, encoding='utf-8')
                        feat = self.boundingbox_feature_dict[x]
                        # a.append(feat)
                        var[i][j] = torch.from_numpy(feat)
                    # var.append(a)
        else:
            raise ValueError("use_vocab and sequential should all be set.")
        # var = np.array(var)
        # var = torch.from_numpy(var)

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var


class Multi30KBoundingBoxDatasetProfile:
    '''
    Note: considered_phrase_type option is different from
    the same name option in datasets.Multi30KBoundingBoxDataset.
    considered_phrase_type here should be set when real_time_parse=True
    But the option in datasets.Multi30KBoundingBoxDataset supports
    offline_parse(when real_time_parse=False).
    '''
    def __init__(self,
                 train_part,
                 val_part,
                 test_part,
                 key_word_pos_list_file,
                 key_word_list_file,
                 batch_size,
                 batch_first=True,
                 include_lengths=True,
                 spacy_nlp=None,
                 real_time_parse=False,
                 considered_phrase_type=None,
                 key_word_own_image=False,
                 average_multiple_images=False,
                 share_vocab=False,
                 max_sentence_length=50,
                 src_sos='<sos>',
                 tgt_sos='<sos>',
                 eos='<eos>',
                 pad='<pad>',
                 unk='<unk>',
                 src='en',
                 tgt='de'):
        self.train_part = train_part
        self.val_part = val_part
        self.test_part = test_part
        self.key_word_pos_list_file = key_word_pos_list_file
        self.key_word_list_file = key_word_list_file
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.include_lengths = include_lengths
        self.spacy_nlp = spacy_nlp
        self.real_time_parse = real_time_parse
        self.considered_phrase_type = considered_phrase_type
        self.key_word_own_image = key_word_own_image
        self.average_multiple_images = average_multiple_images
        self.share_vocab = share_vocab
        self.max_sentence_length = max_sentence_length
        self.src_sos = src_sos
        self.tgt_sos = tgt_sos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.src = src
        self.tgt = tgt

        if self.real_time_parse and self.spacy_nlp is None:
            self.spacy_nlp = spacy.load('en_core_web_sm')

        self.tag_type = 'key_tag' if self.key_word_own_image else 'all_tag'

        self.train_splits_path = self.train_part['splits_path']
        self.train_src_json_path = self.train_part['en_json_path']
        self.train_tgt_path = "%s.%s" % (self.train_part['sentences_prefix'], tgt)
        self.train_image_features_file = self.train_part['boundingbox_features']

        self.val_splits_path = self.val_part['splits_path']
        self.val_src_json_path = self.val_part['en_json_path']
        self.val_tgt_path = "%s.%s" % (self.val_part['sentences_prefix'], tgt)
        self.val_image_features_file = self.val_part['boundingbox_features']

        self.test_splits_path = self.test_part['splits_path']
        self.test_src_json_path = self.test_part['en_json_path']
        self.test_tgt_path = "%s.%s" % (self.test_part['sentences_prefix'], tgt)
        self.test_image_features_file = self.test_part['boundingbox_features']

    def get_part(self, part):
        if part == DATA_PART.Train:
            return {"splits": self.train_splits_path,
                    "src": self.train_src_json_path,
                    "tgt": self.train_tgt_path}
        elif part == DATA_PART.Val:
            return {"splits": self.val_splits_path,
                    "src": self.val_src_json_path,
                    "tgt": self.val_tgt_path}
        elif part == DATA_PART.Test:
            return {"splits": self.test_splits_path,
                    "src": self.test_src_json_path,
                    "tgt": self.test_tgt_path}
        else:
            raise ValueError("Unknown dataset part %s" % part)

    def build_fields(self):
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
        id_field = data.Field(sequential=False, use_vocab=False)
        img_pad_token = -1
        img_field = data.Field(sequential=True, use_vocab=False,
                               init_token=img_pad_token,
                               eos_token=img_pad_token,
                               unk_token=img_pad_token,
                               pad_token=img_pad_token,
                               batch_first=self.batch_first,
                               include_lengths=self.include_lengths)
        # img_field = BBXField(sequential=True, use_vocab=True,
        #                      init_token=img_pad_token, eos_token=img_pad_token,
        #                      pad_token=img_pad_token, unk_token=img_pad_token,
        #                      batch_first=batch_first)
        fields = [("id", id_field), ("src", src_field), ("tgt", tgt_field), ("img", img_field)]
        suffixes = (self.src, self.tgt)
        setattr(self, "fields", fields)
        setattr(self, "suffixes", suffixes)
        return fields, suffixes

# TODO: real_time_parse should be added to torchtext version dataset
    def load_examples(self, part):
        fields = getattr(self, "fields")
        data_parts = self.get_part(part)
        splits_path = data_parts['splits']
        src_json_path = data_parts['src']
        tgt_sent_path = data_parts['tgt']
        with open(splits_path, 'r') as fp:
            splits = [s.strip().split('.')[0] for s in fp.readlines()]
        with open(src_json_path, 'r', encoding='utf-8') as fp:
            src_sentences_json = json.load(fp)
        with open(tgt_sent_path, 'r') as fp:
            tgt_sentences = fp.readlines()

        with open(self.key_word_pos_list_file, 'r') as fp:
            key_word_pos_list = [l.strip() for l in fp.readlines()]
        if self.key_word_list_file is not None:
            with open(self.key_word_list_file, 'r') as fp:
                key_word_list = [l.strip() for l in fp.readlines()]
        else:
            key_word_list = None

        examples = []
        for i, (split, tgt_sent) in enumerate(zip(splits, tgt_sentences)):
            src_json = src_sentences_json[split]
            src_sent = src_json["sentence"]
            if self.real_time_parse:
                src_img_tag = self.__tag_source_sentence__(src_json, key_word_pos_list, key_word_list)
                src_img_tag = [int(t) if t is not None else -1 for t in src_img_tag]
            else:
                src_img_tag = [int(t) if t is not None else -1 for t in src_json[self.tag_type]]
            datalist = [i, src_sent, tgt_sent, src_img_tag]
            ex = data.Example.fromlist(datalist, fields)
            max_len = self.max_sentence_length
            if max_len is None:
                examples.append(ex)
            elif len(ex.src) <= max_len and len(ex.tgt) <= max_len:
                examples.append(ex)
        return examples


    def __tag_source_sentence__(self, src_sentence, key_word_pos_list, key_word_list):
        phrases = sorted(src_sentence['phrases'], key=lambda item: item['first_word_index'])
        nlp = self.spacy_nlp
        sentence = src_sentence['sentence'].strip().split()
        if self.key_word_own_image:
            sentence_doc = nlp(src_sentence['sentence'])
            sentence_pos = conject_postag(src_sentence['sentence'], sentence_doc)
            assert len(sentence) == len(sentence_pos)
        tag = [None] * len(sentence)
        for phrase in phrases:
            first_word_index = phrase['first_word_index']
            phrase_types = phrase['phrase_type']
            if self.considered_phrase_type is not None:
                if len(set(phrase_types) & set(self.considered_phrase_type)) == 0:
                    continue
            phrase_len = len(phrase['phrase'].strip().split())
            for i in range(phrase_len):
                if self.key_word_own_image:
                    if sentence_pos[i + first_word_index] in key_word_pos_list or \
                            (key_word_list is not None and sentence[i + first_word_index] in key_word_list):
                        tag[i + first_word_index] = phrase['phrase_id']
                else:
                    tag[i + first_word_index] = phrase['phrase_id']
        return tag


    def load_extra_data(self, parts):
        if isinstance(parts, DATA_PART):
            parts = [parts]
        for part in parts:
            if part == DATA_PART.Train:
                image_features = np.load(self.train_image_features_file, allow_pickle=True, encoding='bytes')
                setattr(self, "train_images", image_features)
            elif part == DATA_PART.Val:
                image_features = np.load(self.val_image_features_file, allow_pickle=True, encoding='bytes')
                setattr(self, "val_images", image_features)
            elif part == DATA_PART.Test:
                image_features = np.load(self.test_image_features_file, allow_pickle=True, encoding='bytes')
                setattr(self, "test_images", image_features)
            else:
                raise ValueError("Unknown extra data part %s." % part)

    def pack_data(self, data, part):
        id = data.id
        src, src_len = data.src
        tgt, tgt_len = data.tgt
        img, img_len = data.img
        assert all(src_len == img_len)
        if part == DATA_PART.Train:
            image_features = getattr(self, "train_images")
        elif part == DATA_PART.Val:
            image_features = getattr(self, "val_images")
        elif part == DATA_PART.Test:
            image_features = getattr(self, "test_images")
        else:
            raise ValueError("Unknown extra data part %s." % part)

        batch_size = img.shape[0] if self.batch_first else img.shape[1]
        max_seq_len = img.shape[1] if self.batch_first else img.shape[0]
        feat_dim = image_features[0][list(image_features[0].keys())[0]][0].shape[-1]

        img = img if self.batch_first else img.transpose(0,1)
        src_img = torch.zeros([batch_size, max_seq_len - 1, feat_dim])
        feat_mask = torch.zeros([batch_size, max_seq_len - 1])
        for i, (sent_id, img_seq) in enumerate(zip(id, img)):
            for j, phrase_id in enumerate(img_seq[:-1]):
                if phrase_id < 0:
                    continue
                phrase_id = bytes(str(phrase_id.item()), encoding='utf-8')
                if not phrase_id in image_features[sent_id]:
                    continue
                if self.average_multiple_images:
                    feat = np.average(image_features[sent_id][phrase_id], axis=0)
                else:
                    # print(img)
                    # print(img_seq)
                    # print(sent_id, image_features[sent_id].keys())
                    feat = image_features[sent_id][phrase_id][0]

                src_img[i][j] = torch.from_numpy(feat)
                feat_mask[i][j] = 1.0

        if self.batch_first:
            sos_src = src[:, :-1]
            eos_src = src[:, 1:]
            sos_tgt = tgt[:, :-1]
            eos_tgt = tgt[:, 1:]
        else:
            sos_src = src[:-1]
            eos_src = src[1:]
            sos_tgt = tgt[:-1]
            eos_tgt = tgt[1:]
        src_len = src_len.unsqueeze(1) - 1
        tgt_len = tgt_len.unsqueeze(1) - 1
        src_img = src_img if self.batch_first else src_img.transpose(0, 1)
        img_mask = feat_mask if self.batch_first else feat_mask.transpose(0, 1)

        return [sos_src, eos_src,
                sos_tgt, eos_tgt,
                src_len, tgt_len,
                src_img, img_mask]

    # sos_src = packed_data[0].to(device)
    # eos_src = packed_data[1].to(device)
    # sos_tgt = packed_data[2].to(device)
    # eos_tgt = packed_data[3].to(device)
    # src_len = packed_data[4].squeeze(1).to(device)
    # tgt_len = packed_data[5].squeeze(1).to(device)
    # src_img = packed_data[6].to(device)
    # img_mask = packed_data[7].to(device)
