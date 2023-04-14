# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:multi30k_profile.py
@time:2020-08-29 
@desc:
'''
import os
from utils.config import DATA_PART
from utils import utils


class Multi30kProfile:
    def __init__(self):
        self.train_text_embeddings_file = None
        self.val_text_embeddings_file = None
        self.test_text_embeddings_file = None

        self.train_avgimage_features_file = None
        self.val_avgimage_features_file = None
        self.test_avgimage_features_file = None

        self.vocab_with_image_file = None
        self.average_image_features_file = None
        self.key_word_pos_list_file = None
        self.key_word_list_file = None

        self.train_image_file_global = None
        self.val_image_file_global = None
        self.test_image_file_global = None

        self.train_image_file_local = None
        self.val_image_file_local = None
        self.test_image_file_local = None

        self.train_boundingbox_features_file = None
        self.val_boundingbox_features_file = None
        self.test_boundingbox_features_file = None

        self.train_splits_file = None
        self.val_splits_file = None
        self.test_splits_file = None

        self.train_source_json_file = None
        self.val_source_json_file = None
        self.test_source_json_file = None

        self.train_prefix = None
        self.val_prefix = None
        self.test_prefix = None
        self.vocab_prefix = None

        self.train_ref_prefix = None
        self.val_ref_prefix = None
        self.test_ref_prefix = None

        self.languages = None

    def get_text_embedding_file(self, part):
        if part == DATA_PART.Train:
            return self.train_text_embeddings_file
        elif part == DATA_PART.Val:
            return self.val_text_embeddings_file
        elif part == DATA_PART.Test:
            return self.test_text_embeddings_file
        else:
            raise ValueError("Unknown dataset part %s" % part)

    def get_avgimage_features_file(self, part):
        if part == DATA_PART.Train:
            return self.train_avgimage_features_file
        elif part == DATA_PART.Val:
            return self.val_avgimage_features_file
        elif part == DATA_PART.Test:
            return self.test_avgimage_features_file
        else:
            raise ValueError("Unknown dataset part %s" % part)

    def get_prefix(self, part, is_reference=False):
        if part == DATA_PART.Train:
            return self.train_prefix if not is_reference else self.train_ref_prefix
        elif part == DATA_PART.Val:
            return self.val_prefix if not is_reference else self.val_ref_prefix
        elif part == DATA_PART.Test:
            return self.test_prefix if not is_reference else self.test_ref_prefix
        elif part == DATA_PART.Vocab:
            return self.vocab_prefix
        else:
            raise ValueError("Unknown dataset part %s" % part)

    def get_file(self, part, suffix, counted=False):
        if not suffix in self.languages:
            raise ValueError("Unknown language %s" % suffix)
        pre = self.get_prefix(part)
        if counted:
            return "%s.counted.%s" % (pre, suffix)
        return "%s.%s" % (pre, suffix)

    def get_attributes(self, attr_list):
        ret = {}
        for attr in attr_list:
            val = getattr(self, attr)
            ret[attr] = val
        return ret

    def get_share_vocab_file(self, languages, counted=False):
        assert len(languages) == 2
        vocab_prefix = "%s.counted" % self.vocab_prefix if counted else self.vocab_prefix
        lang = '-'.join(languages)
        reverse_lang = '-'.join([languages[1], languages[0]])
        vocab_path = "%s.%s" % (vocab_prefix, lang)
        reverse_vocab_path = "%s.%s" % (vocab_prefix, reverse_lang)
        if os.path.exists(vocab_path):
            return vocab_path
        elif os.path.exists(reverse_vocab_path):
            return reverse_vocab_path
        else:
            raise ValueError("Could not find vocabulary file at %s or %s" % (vocab_path, reverse_vocab_path))

    def get_splits_file(self, part):
        if part == DATA_PART.Train:
            return self.train_splits_file
        elif part == DATA_PART.Val:
            return self.val_splits_file
        elif part == DATA_PART.Test:
            return self.test_splits_file
        else:
            raise ValueError("Unknown dataset part %s" % part)

    def get_json_source_file(self, part):
        if part == DATA_PART.Train:
            return self.train_source_json_file
        elif part == DATA_PART.Val:
            return self.val_source_json_file
        elif part == DATA_PART.Test:
            return self.test_source_json_file
        else:
            raise ValueError("Unknown dataset part %s" % part)

    def get_image_feature_file(self, part, suffix):
        if not suffix in ['global', 'local', 'boundingbox']:
            raise ValueError("Unknown image feature %s" % suffix)
        if part == DATA_PART.Train:
            if suffix == 'global':
                return self.train_image_file_global
            elif suffix == 'local':
                return self.train_image_file_local
            else:
                return self.train_boundingbox_features_file
        elif part == DATA_PART.Val:
            if suffix == 'global':
                return self.val_image_file_global
            elif suffix == 'local':
                return self.val_image_file_local
            else:
                return self.val_boundingbox_features_file
        elif part == DATA_PART.Test:
            if suffix == 'global':
                return self.test_image_file_global
            elif suffix == 'local':
                return self.test_image_file_local
            else:
                return self.test_boundingbox_features_file
        else:
            raise ValueError("Unknown dataset part %s" % part)

    def get_reference_file(self, part: DATA_PART, lang):
        if part == DATA_PART.Train:
            if self.train_ref_prefix:
                return "%s.%s" % (self.train_ref_prefix, lang)
        elif part == DATA_PART.Test:
            if self.test_ref_prefix:
                return "%s.%s" % (self.test_ref_prefix, lang)
        elif part == DATA_PART.Val:
            if self.val_ref_prefix:
                return "%s.%s" % (self.val_ref_prefix, lang)
        else:
            raise ValueError("Unknown dataset part %s" % part)
        return self.get_file(part, lang)

    def get_part(self, part: DATA_PART):
        return {'sentences_prefix': self.get_prefix(part),
                'reference_prefix': self.get_prefix(part, is_reference=True),
                'global_features': self.get_image_feature_file(part, 'global'),
                'local_features': self.get_image_feature_file(part, 'local'),
                'boundingbox_features': self.get_image_feature_file(part, 'boundingbox'),
                'en_json_path': self.get_json_source_file(part),
                'splits_path': self.get_splits_file(part)}


    def load_config(self, config_path):
        import json
        with open(config_path, 'r') as fp:
            notes_removed_config_str = utils.remove_json_notes(fp)
            external_config = json.loads(notes_removed_config_str)
        self.__dict__.update(external_config)
