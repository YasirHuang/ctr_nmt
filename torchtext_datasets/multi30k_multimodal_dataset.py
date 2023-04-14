# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:multi30k_multimodal_dataset.py
@time:2020-09-27 
@desc:
'''
import numpy as np

import torch
from torchtext_datasets.multi30k_text_dataset import Multi30KTextDatasetProfile

from utils.config import DATA_PART


class Multi30kMultimodalDatasetProfile(Multi30KTextDatasetProfile):

    def __init__(self,
                 src,
                 tgt,
                 train_part,
                 val_part,
                 test_part,
                 feature_type,  # global, local, both or none
                 batch_size,
                 batch_first=True,
                 include_lengths=True,
                 share_vocab=False,
                 max_sentence_length=50,
                 src_sos='<sos>',
                 tgt_sos='<sos>',
                 eos='<eos>',
                 pad='<pad>',
                 unk='<unk>'):

        self.train_prefix = train_part['sentences_prefix']
        self.train_global_features_path = train_part['global_features']
        self.train_local_features_path = train_part['local_features']
        self.val_prefix = val_part['sentences_prefix']
        self.val_global_features_path = val_part['global_features']
        self.val_local_features_path = val_part['local_features']
        self.test_prefix = test_part['sentences_prefix']
        self.test_global_features_path = test_part['global_features']
        self.test_local_features_path = test_part['local_features']

        self.feature_type = feature_type
        self.use_global_features = False
        self.use_local_features = False
        if self.feature_type == 'global':
            self.use_global_features = True
        elif self.feature_type == 'local':
            self.use_local_features = True
        elif self.feature_type == 'both':
            self.use_global_features = True
            self.use_local_features = True
        elif self.feature_type == 'none':
            pass
        else:
            raise ValueError('Invalid feature type %s' % self.feature_type)

        super(Multi30kMultimodalDatasetProfile, self).__init__(
            src,
            tgt,
            self.train_prefix,
            self.val_prefix,
            self.test_prefix,
            batch_size,
            batch_first=batch_first,
            include_lengths=include_lengths,
            share_vocab=share_vocab,
            max_sentence_length=max_sentence_length,
            src_sos=src_sos,
            tgt_sos=tgt_sos,
            eos=eos,
            pad=pad,
            unk=unk)

    def load_extra_data(self, parts):
        if isinstance(parts, DATA_PART):
            parts = [parts]
        for part in parts:
            if part == DATA_PART.Train:
                if self.use_global_features:
                    g = np.load(self.train_global_features_path)
                    setattr(self, "train_global_images", g)
                if self.use_local_features:
                    l = np.load(self.train_local_features_path)
                    setattr(self, "train_local_images", l)
            elif part == DATA_PART.Val:
                if self.use_global_features:
                    g = np.load(self.val_global_features_path)
                    setattr(self, "val_global_images", g)
                if self.use_local_features:
                    l = np.load(self.val_local_features_path)
                    setattr(self, "val_local_images", l)
            elif part == DATA_PART.Test:
                if self.use_global_features:
                    g = np.load(self.test_global_features_path)
                    setattr(self, "test_global_images", g)
                if self.use_local_features:
                    l = np.load(self.test_local_features_path)
                    setattr(self, "test_local_images", l)
            else:
                raise ValueError("Unknown extra data part %s." % part)

    def get_part(self, part):
        if part == DATA_PART.Train:
            l = getattr(self, 'train_local_images') if self.use_local_features else None
            g = getattr(self, 'train_global_images') if self.use_global_features else None
        elif part == DATA_PART.Val:
            l = getattr(self, 'val_local_images') if self.use_local_features else None
            g = getattr(self, 'val_global_images') if self.use_global_features else None
        elif part == DATA_PART.Test:
            l = getattr(self, 'test_local_images') if self.use_local_features else None
            g = getattr(self, 'test_global_images') if self.use_global_features else None
        else:
            raise ValueError("Unknown extra data part %s." % part)
        return l, g

    def pack_data(self, data, part):
        id = data.id
        ret = super(Multi30kMultimodalDatasetProfile, self).pack_data(data, part)
        local_feats, global_feats = self.get_part(part)
        if self.use_global_features:
            global_feat = torch.from_numpy(global_feats[id])
            ret.append(global_feat)
        if self.use_local_features:
            local_feat = torch.from_numpy(local_feats[id])
            batch_size = local_feat.shape[0]
            seq_len = local_feat.shape[1]
            local_feat_len = torch.full((batch_size,1), seq_len).int()
            ret.append(local_feat)
            ret.append(local_feat_len)

        return ret

        # sos_src = packed_data[0].to(device)
        # eos_src = packed_data[1].to(device)
        # sos_tgt = packed_data[2].to(device)
        # eos_tgt = packed_data[3].to(device)
        # src_len = packed_data[4].squeeze(1).to(device)
        # tgt_len = packed_data[5].squeeze(1).to(device)
        # local_feat = packed_data[6].to(device)
        # local_feat_len = packed_data[7].squeeze(1).to(device)
