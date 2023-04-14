# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:dataset_manager.py
@time:2020-08-29 
@desc:
'''
import json

from datasets.multi30k_profile import Multi30kProfile
from datasets.multi30k_text_dataset import build_dataloader as build_multi30k_dataloader
from datasets.multi30k_multimodal_dataset import build_dataloader as build_mm_multi30k_dataloader
from datasets.multi30k_boundingbox_dataset import build_dataloader as build_bb_multi30k_dataloader
from utils import utils

from utils.config import DATA_PART


class DatasetManager:
    def __init__(self,
                 FLAGS,
                 which_part
                 ):
        '''
        Now registered datasets include:
        "multi30k_text",
        "multi30k_global_feature",
        "multi30k_local_feature",
        "multi30k_boundingbox_feature",
        "multi30k_avgboundingbox_feature"
        :param FLAGS:
        :param which_part: DATA_PART.Train or DATA_PART.Val or DATA_PART.Test
        '''

        self.which_part = which_part

        self.dataset = FLAGS.dataset
        self.dataset_config_file = FLAGS.dataset_config_file
        self.share_vocab = FLAGS.share_vocab
        self.src = FLAGS.src
        self.tgt = FLAGS.tgt
        self.batch_size = FLAGS.batch_size
        self.min_sentence_length = FLAGS.min_len
        self.max_sentence_length = FLAGS.max_len
        self.shuffle = FLAGS.shuffle_dataset
        self.num_workers = FLAGS.num_workers
        self.no_vocab = FLAGS.no_vocab
        self.individual_start_token = FLAGS.individual_start_token
        self.sos = FLAGS.sos
        self.eos = FLAGS.eos
        self.unk = FLAGS.unk
        self.pad = FLAGS.pad
        self.special_tokens = FLAGS.special_tokens
        self.real_time_parse = FLAGS.real_time_parse
        self.considered_phrase_type = FLAGS.considered_phrase_type
        self.key_word_own_image = FLAGS.key_word_own_image
        self.average_multiple_images = FLAGS.average_multiple_images

        self.noise_type = FLAGS.noise_type
        self.placeholder_id = FLAGS.placeholder_id
        self.uniform_noise_dim = FLAGS.uniform_noise_dim
        self.random_place_ratio = FLAGS.random_place_ratio

        self.src_sos = self.sos
        self.tgt_sos = self.sos
        # prepare dataset profile
        if 'multi30k' in self.dataset.lower():
            dp = Multi30kProfile()
            dp.load_config(self.dataset_config_file)

            if self.share_vocab:
                if self.individual_start_token:
                    self.src_sos = "%s%s" % (self.src, self.sos)
                    self.tgt_sos = "%s%s" % (self.tgt, self.sos)
                    init_dict = {self.pad: 0, self.eos: 1, self.src_sos: 2, self.tgt_sos: 3, self.unk: 4}
                else:
                    init_dict = {self.pad: 0, self.eos: 1, self.sos: 2, self.unk: 3}

                word_to_id_dict, vocab_size = utils.build_word2iddict(dp.get_share_vocab_file([self.src, self.tgt]),
                                                                      initial_dict=init_dict,
                                                                      special_tokens=self.special_tokens)
                tgt_id_to_word_dict, _ = utils.build_id2worddict(dp.get_share_vocab_file([self.src, self.tgt]),
                                                                 word_to_id_dict)
                src_id_to_word_dict = tgt_id_to_word_dict
                src_word_to_id_dict = word_to_id_dict
                tgt_word_to_id_dict = word_to_id_dict
                src_vocab_size = vocab_size
                tgt_vocab_size = vocab_size
                print("Shared vocabulary has builded, vocab_size is %d" % vocab_size)
            elif self.no_vocab:
                src_word_to_id_dict = None
                tgt_word_to_id_dict = None
                src_id_to_word_dict = None
                tgt_id_to_word_dict = None
                src_vocab_size = 0
                tgt_vocab_size = 0
            else:
                src_word_to_id_dict, src_vocab_size = utils.build_word2iddict(dp.get_file(DATA_PART.Vocab, self.src),
                                                                              initial_dict={self.pad: 0,
                                                                                            self.eos: 1,
                                                                                            self.sos: 2,
                                                                                            self.unk: 3},
                                                                              special_tokens=self.special_tokens)
                src_id_to_word_dict, _ = utils.build_id2worddict(dp.get_file(DATA_PART.Vocab, self.tgt),
                                                                 src_word_to_id_dict)

                print('Source vocabulary has builded, vocab_size is %d' % src_vocab_size)
                tgt_word_to_id_dict, tgt_vocab_size = utils.build_word2iddict(dp.get_file(DATA_PART.Vocab, self.tgt),
                                                                              initial_dict={self.pad: 0,
                                                                                            self.eos: 1,
                                                                                            self.sos: 2,
                                                                                            self.unk: 3},
                                                                              special_tokens=self.special_tokens)
                tgt_id_to_word_dict, _ = utils.build_id2worddict(dp.get_file(DATA_PART.Vocab, self.tgt),
                                                                 tgt_word_to_id_dict)
                print('Target vocabulary has builded, vocab_size is %d' % tgt_vocab_size)
        else:
            raise ValueError("Unknown dataset.")

        # prepare dataset loaders
        if which_part is None:
            which_part = [DATA_PART.Train, DATA_PART.Val, DATA_PART.Test]
        elif isinstance(which_part, DATA_PART):
            which_part = [which_part]

        iterable_datas = {}
        if self.dataset == 'multi30k' or self.dataset == 'multi30k_text':
            for part in which_part:
                dataloader = build_multi30k_dataloader(dp.get_file(part, self.src),
                                                       dp.get_file(part, self.tgt),
                                                       src_word_to_id_dict,
                                                       tgt_word_to_id_dict,
                                                       self.batch_size,
                                                       min_sentence_length=self.min_sentence_length,
                                                       max_sentence_length=self.max_sentence_length,
                                                       shuffle=self.shuffle if part == DATA_PART.Train else False,
                                                       num_workers=self.num_workers,
                                                       src_sos=self.src_sos,
                                                       tgt_sos=self.tgt_sos,
                                                       eos=self.eos,
                                                       unk=self.unk)

                iterable_datas[part] = dataloader
        elif self.dataset == 'multi30k_global_feature':
            for part in which_part:
                dataloader = build_mm_multi30k_dataloader(dp.get_file(part, self.src),
                                                          dp.get_file(part, self.tgt),
                                                          dp.get_image_feature_file(part, 'global'),
                                                          None,
                                                          src_word_to_id_dict,
                                                          tgt_word_to_id_dict,
                                                          self.batch_size,
                                                          min_sentence_length=self.min_sentence_length,
                                                          max_sentence_length=self.max_sentence_length,
                                                          shuffle=self.shuffle if part == DATA_PART.Train else False,
                                                          num_workers=self.num_workers,
                                                          src_sos=self.src_sos,
                                                          tgt_sos=self.tgt_sos,
                                                          eos=self.eos,
                                                          unk=self.unk,
                                                          load_global_features=True,
                                                          load_local_features=False)
                iterable_datas[part] = dataloader
        elif self.dataset == 'multi30k_local_feature':
            for part in which_part:
                dataloader = build_mm_multi30k_dataloader(dp.get_file(part, self.src),
                                                          dp.get_file(part, self.tgt),
                                                          None,
                                                          dp.get_image_feature_file(part, 'local'),
                                                          src_word_to_id_dict,
                                                          tgt_word_to_id_dict,
                                                          self.batch_size,
                                                          min_sentence_length=self.min_sentence_length,
                                                          max_sentence_length=self.max_sentence_length,
                                                          shuffle=self.shuffle if part == DATA_PART.Train else False,
                                                          num_workers=self.num_workers,
                                                          src_sos=self.src_sos,
                                                          tgt_sos=self.tgt_sos,
                                                          eos=self.eos,
                                                          unk=self.unk,
                                                          load_global_features=False,
                                                          load_local_features=True)
                iterable_datas[part] = dataloader
        elif self.dataset == 'multi30k_boundingbox_feature':
            for part in which_part:
                dataloader = build_bb_multi30k_dataloader(dp.get_splits_file(part),
                                                          dp.get_json_source_file(part),
                                                          dp.get_file(part, self.tgt),
                                                          src_word_to_id_dict,
                                                          tgt_word_to_id_dict,
                                                          dp.get_image_feature_file(part, 'boundingbox'),
                                                          dp.key_word_pos_list_file,
                                                          dp.key_word_list_file,
                                                          self.batch_size,
                                                          real_time_parse=self.real_time_parse,
                                                          considered_phrase_type=self.considered_phrase_type,
                                                          key_word_own_image=self.key_word_own_image,
                                                          average_multiple_images=self.average_multiple_images,
                                                          shuffle=self.shuffle if part == DATA_PART.Train else False,
                                                          num_workers=self.num_workers,
                                                          src_sos=self.src_sos,
                                                          tgt_sos=self.tgt_sos,
                                                          eos=self.eos,
                                                          unk=self.unk)
                iterable_datas[part] = dataloader
        else:
            raise ValueError("Unknown dataset %s" % self.dataset)

        self.data = iterable_datas
        self.dataset_profile = dp
        self.src_word_to_id_dict = src_word_to_id_dict
        self.tgt_word_to_id_dict = tgt_word_to_id_dict
        self.src_id_to_word_dict = src_id_to_word_dict
        self.tgt_id_to_word_dict = tgt_id_to_word_dict
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        FLAGS.src_vocab_size = src_vocab_size
        FLAGS.tgt_vocab_size = tgt_vocab_size

        def get_word_id(vocab, word):
            if vocab is not None:
                return vocab[word]
            else:
                return None

        FLAGS.pad_token_id = get_word_id(src_word_to_id_dict, self.pad)
        FLAGS.src_sos_token_id = get_word_id(src_word_to_id_dict, self.src_sos)
        FLAGS.tgt_sos_token_id = get_word_id(src_word_to_id_dict, self.tgt_sos)
        FLAGS.eos_token_id = get_word_id(src_word_to_id_dict, self.eos)
        FLAGS.unk = get_word_id(src_word_to_id_dict, self.unk)

    def get_dataloader(self, part=DATA_PART.Train):
        return self.data[part]


def test_dataset_manager(FLAGS):
    which_part = [DATA_PART.Train, DATA_PART.Val, DATA_PART.Test]
    dm = DatasetManager(FLAGS, which_part)
    dl = dm.get_dataloader(DATA_PART.Train)
    for data in dl:
        print(data)
