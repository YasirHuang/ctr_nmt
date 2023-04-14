# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:torchtext_dataset_manager.py
@time:2020-09-19 
@desc:
'''
import os
from collections import Counter
import logging

import torch
from torchtext import data
from torchtext.vocab import Vocab

from torchtext_datasets.parallel_dataset import NormalDataset
from torchtext_datasets.iterators import OrderedIterator
from datasets.multi30k_profile import Multi30kProfile
from torchtext_datasets.utils import dyn_batch_with_padding

from torchtext_datasets import multi30k_text_dataset as m30k_t
from torchtext_datasets import multi30k_multimodal_dataset as m30k_mm
from torchtext_datasets import multi30k_boundingbox_dataset as m30k_bbx

from utils.config import DATA_PART


class DatasetManager:
    def __init__(self,
                 FLAGS,
                 which_part
                 ):

        self.which_part = which_part

        self.iterator = FLAGS.iterator
        self.repeated_iterator = FLAGS.repeated_iterator
        self.dataset = FLAGS.dataset
        self.dataset_config_file = FLAGS.dataset_config_file
        self.share_vocab = FLAGS.share_vocab
        self.max_vocab_size = FLAGS.max_vocab_size
        self.load_vocab = FLAGS.load_vocab
        self.vocab_path = FLAGS.vocab_path
        self.out_dir = FLAGS.out_dir
        self.src = FLAGS.src
        self.tgt = FLAGS.tgt
        self.batch_size = FLAGS.batch_size
        self.eval_batch_size = FLAGS.eval_batch_size
        self.infer_batch_size = FLAGS.infer_batch_size
        self.batch_first = FLAGS.batch_first
        self.min_sentence_length = FLAGS.min_len
        self.max_sentence_length = FLAGS.max_len
        self.shuffle = FLAGS.shuffle_dataset
        self.num_workers = FLAGS.num_workers
        self.individual_start_token = FLAGS.individual_start_token
        self.sos = FLAGS.sos
        self.eos = FLAGS.eos
        self.pad = FLAGS.pad
        self.unk = FLAGS.unk
        self.real_time_parse = FLAGS.real_time_parse
        self.considered_phrase_type = FLAGS.considered_phrase_type
        self.key_word_own_image = FLAGS.key_word_own_image
        self.average_multiple_images = FLAGS.average_multiple_images

        self.noise_type = FLAGS.noise_type
        self.uniform_noise_dim = FLAGS.uniform_noise_dim

        self.special_tokens = FLAGS.special_tokens


        self.src_sos = self.sos
        self.tgt_sos = self.sos
        if self.individual_start_token:
            if not self.share_vocab:
                logging.warning('"share_vocab" should be set when "individual_start_token" was set.')
            self.src_sos = "%s%s" % (self.src, self.sos)
            self.tgt_sos = "%s%s" % (self.tgt, self.sos)
        dp = self.load_dataset_profile(self.dataset, self.dataset_config_file)
        # prepare dataset loaders
        if which_part is None:
            which_part = [DATA_PART.Train, DATA_PART.Val, DATA_PART.Test]
        elif isinstance(which_part, DATA_PART):
            which_part = [which_part]

        if self.dataset == 'multi30k' or self.dataset == 'multi30k_text':
            dataset_profile = m30k_t.Multi30KTextDatasetProfile(
                self.src,
                self.tgt,
                dp.get_prefix(DATA_PART.Train),
                dp.get_prefix(DATA_PART.Val),
                dp.get_prefix(DATA_PART.Test),
                self.batch_size,
                batch_first=self.batch_first,
                include_lengths=True,
                share_vocab=self.share_vocab,
                max_sentence_length=self.max_sentence_length,
                src_sos=self.src_sos,
                tgt_sos=self.tgt_sos,
                eos=self.eos,
                pad=self.pad,
                unk=self.unk)

        elif self.dataset == 'multi30k_global_feature':
            dataset_profile = m30k_mm.Multi30kMultimodalDatasetProfile(
                src=self.src,
                tgt=self.tgt,
                train_part=dp.get_part(DATA_PART.Train),
                val_part=dp.get_part(DATA_PART.Val),
                test_part=dp.get_part(DATA_PART.Test),
                feature_type='global',  # global, local, both or none
                batch_size=self.batch_size,
                batch_first=self.batch_first,
                include_lengths=True,
                share_vocab=self.share_vocab,
                max_sentence_length=self.max_sentence_length,
                src_sos=self.src_sos,
                tgt_sos=self.tgt_sos,
                eos=self.eos,
                pad=self.pad,
                unk=self.unk
            )
        elif self.dataset == 'multi30k_local_feature':
            dataset_profile = m30k_mm.Multi30kMultimodalDatasetProfile(
                src=self.src,
                tgt=self.tgt,
                train_part=dp.get_part(DATA_PART.Train),
                val_part=dp.get_part(DATA_PART.Val),
                test_part=dp.get_part(DATA_PART.Test),
                feature_type='local',  # global, local, both or none
                batch_size=self.batch_size,
                batch_first=self.batch_first,
                include_lengths=True,
                share_vocab=self.share_vocab,
                max_sentence_length=self.max_sentence_length,
                src_sos=self.src_sos,
                tgt_sos=self.tgt_sos,
                eos=self.eos,
                pad=self.pad,
                unk=self.unk
            )
        elif self.dataset == 'multi30k_boundingbox_feature':
            dataset_profile = m30k_bbx.Multi30KBoundingBoxDatasetProfile(
                train_part=dp.get_part(DATA_PART.Train),
                val_part=dp.get_part(DATA_PART.Val),
                test_part=dp.get_part(DATA_PART.Test),
                key_word_pos_list_file=dp.key_word_pos_list_file,
                key_word_list_file=dp.key_word_list_file,
                batch_size=self.batch_size,
                batch_first=self.batch_first,
                include_lengths=True,
                spacy_nlp=None,
                real_time_parse=self.real_time_parse,
                considered_phrase_type=self.considered_phrase_type,
                key_word_own_image=self.key_word_own_image,
                average_multiple_images=self.average_multiple_images,
                share_vocab=self.share_vocab,
                max_sentence_length=self.max_sentence_length,
                src_sos=self.src_sos,
                tgt_sos=self.tgt_sos,
                eos=self.eos,
                pad=self.pad,
                unk=self.unk,
                src=self.src,
                tgt=self.tgt)
        else:
            raise ValueError("Unknown dataset %s" % self.dataset)

        fields, suffixes = dataset_profile.build_fields()

        # load_
        if hasattr(dataset_profile, 'load_extra_data'):
            dataset_profile.load_extra_data(which_part)

        datasets = {}
        for part in which_part:
            assert isinstance(part, DATA_PART)
            examples = dataset_profile.load_examples(part)
            d = NormalDataset(examples, fields)
            datasets[part] = d

        assert fields[1][0] == 'src'
        assert fields[2][0] == 'tgt'
        src_field, tgt_field = fields[1][1], fields[2][1]

        # load or create vocab
        if self.load_vocab:
            assert self.vocab_path is not None
            vocabs = self.load_vocab_from_path(self.vocab_path)
            assert vocabs is not None, "No vocabulary found in %s" % self.vocab_path
            assert isinstance(vocabs, (list, tuple))
            assert isinstance(vocabs[0], Vocab) and isinstance(vocabs[1], Vocab)
            src_field.vocab, tgt_field.vocab = vocabs
        else:
            self.vocab_path = os.path.join(self.out_dir, 'vocab.%s-%s.pt' % (self.src, self.tgt))
            vocabs = self.load_vocab_from_path(self.vocab_path)
            if vocabs is not None:
                print("load existing vocab:", )
                src_field.vocab, tgt_field.vocab = vocabs
                print(len(src_field.vocab), src_field.vocab.itos[:10])
            else:
                if self.individual_start_token:
                    if self.share_vocab:
                        src_vocab = self.load_vocab_from_path(dp.get_share_vocab_file([self.src, self.tgt], True))
                        tgt_vocab = self.load_vocab_from_path(dp.get_share_vocab_file([self.src, self.tgt], True))
                    else:
                        src_vocab = self.load_vocab_from_path(dp.get_file(DATA_PART.Vocab, self.src, counted=True))
                        tgt_vocab = self.load_vocab_from_path(dp.get_file(DATA_PART.Vocab, self.tgt, counted=True))
                    src_field.build_vocab(src_vocab)
                    tgt_field.build_vocab(tgt_vocab)
                else:
                    if self.share_vocab:
                        assert src_field is tgt_field, \
                            "src_field and tgt_field should be the same object, " \
                            "when share_vocab and not individual_start_token"
                        vocab = self.load_vocab_from_path(dp.get_share_vocab_file([self.src, self.tgt], True))
                        src_field.build_vocab(vocab)
                    else:
                        src_vocab = self.load_vocab_from_path(dp.get_file(DATA_PART.Vocab, self.src, counted=True))
                        tgt_vocab = self.load_vocab_from_path(dp.get_file(DATA_PART.Vocab, self.tgt, counted=True))
                        src_field.build_vocab(src_vocab)
                        tgt_field.build_vocab(tgt_vocab)
                self.save_vocab(src_field, tgt_field, self.vocab_path)

        # if self.load_vocab:
        #     assert self.vocab_path is not None
        #     loaded = self.load_vocab_from_path(src_field, tgt_field, self.vocab_path)
        #     if not loaded: raise ValueError("Vocab file not found.")
        # else:
        #     if self.vocab_path is None:
        #         self.vocab_path = os.path.join(self.out_dir, 'vocab.%s-%s.pt' % (self.src, self.tgt))
        #     loaded = self.load_vocab_from_path(src_field, tgt_field, self.vocab_path)
        #     if not loaded:
        #         assert DATA_PART.Train in which_part, "Train part not found in which_part"
        #         src_field.build_vocab(datasets[DATA_PART.Train], max_size=self.max_vocab_size)
        #         if not self.share_vocab:
        #             tgt_field.build_vocab(datasets[DATA_PART.Train], max_size=self.max_vocab_size)
        #         self.save_vocab(src_field, tgt_field, self.vocab_path)

        src_vocab_size = len(src_field.vocab)
        tgt_vocab_size = len(tgt_field.vocab)
        FLAGS.src_vocab_size = src_vocab_size
        FLAGS.tgt_vocab_size = tgt_vocab_size
        FLAGS.pad_token_id = src_field.vocab.stoi[self.pad]
        FLAGS.src_sos_token_id = src_field.vocab.stoi[self.src_sos]
        FLAGS.tgt_sos_token_id = src_field.vocab.stoi[self.tgt_sos]
        FLAGS.eos_token_id = src_field.vocab.stoi[self.eos]
        FLAGS.unk_token_id = src_field.vocab.stoi[self.unk]

        iterable_datas = {}
        for part in which_part:
            if part == DATA_PART.Train:
                if self.iterator.lower() == "bucket" or self.iterator.lower() == "bucketiterator":
                    iterator = data.BucketIterator(datasets[part],
                                                   self.batch_size,
                                                   batch_size_fn=dyn_batch_with_padding,
                                                   train=True,
                                                   repeat=self.repeated_iterator,
                                                   shuffle=True,
                                                   sort_within_batch=True,
                                                   sort=False)
                elif self.iterator.lower() == "ordered" or self.iterator.lower() == "orderediterator":
                    iterator = OrderedIterator(datasets[part],
                                               batch_size=self.batch_size,
                                               batch_size_fn=None,
                                               train=True,
                                               sort=False,
                                               sort_within_batch=True,
                                               repeat=False
                                               )
                else:
                    iterator = data.Iterator(datasets[part],
                                             self.batch_size,
                                             train=True,
                                             repeat=self.repeated_iterator,
                                             shuffle=True,
                                             sort_within_batch=True,
                                             sort=False)
            else:
                iterator = data.Iterator(datasets[part],
                                         self.infer_batch_size,
                                         train=False,
                                         repeat=False,
                                         shuffle=False,
                                         sort=False)
            iterable_datas[part] = iterator
        self.data = iterable_datas
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.dataset_profile = dp
        # dict(stoi/itos) is necessary, when unkonwn word appears in dataset,
        # stoi(collections.defaultdict) will add it in as stoi[the_unknown_word]=0,
        # in case the length of stoi will changed.
        # so we copy it at first
        self.src_word_to_id_dict = dict(src_field.vocab.stoi)
        self.tgt_word_to_id_dict = dict(tgt_field.vocab.stoi)
        self.src_id_to_word_dict = list(src_field.vocab.itos)
        self.tgt_id_to_word_dict = list(tgt_field.vocab.itos)
        self.task_dataset_profile = dataset_profile
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

    def get_dataloader(self, part=DATA_PART.Train):
        return self.data[part]

    def pack_data(self, data, part):
        return self.task_dataset_profile.pack_data(data, part)

    def load_dataset_profile(self, dataset, dataset_config_file):
        if 'multi30k' in dataset.lower():
            dp = Multi30kProfile()
            dp.load_config(dataset_config_file)
        else:
            raise ValueError("Unknown dataset.")
        return dp

    def load_vocab_from_path(self, vocab_path):
        if not os.path.exists(vocab_path):
            return None
        if vocab_path.endswith('pt'):
            vocab = torch.load(str(vocab_path))
            return vocab
        else:
            with open(vocab_path, 'r') as fp:
                lines = fp.readlines()
            counter = Counter(self.special_tokens)
            for l in lines:
                w, n = l.strip().split()
                counter[w] = int(n)
            return counter

    # def load_vocab_from_path(self, src_field, tgt_field, vocab_path):
    #     if vocab_path.endswith('pt'):
    #         if not os.path.exists(vocab_path):
    #             return False
    #         src_vocab, tgt_vocab = torch.load(str(vocab_path))
    #         src_field.vocab = src_vocab
    #         tgt_field.vocab = tgt_vocab
    #     else:
    #         with open(vocab_path, 'r') as fp:
    #             lines = fp.readlines()
    #         counter = dict()
    #         for l in lines:
    #             w, n = l.strip().split()
    #             counter[w] = int(n)
    #         counter = Counter(counter)
    #     return True

    def save_vocab(self, src_field, tgt_field, vocab_path):
        dirname = os.path.dirname(vocab_path)
        if not os.path.exists(dirname):
            os.path.mkdir(dirname)
        torch.save([src_field.vocab, tgt_field.vocab], vocab_path)


def test_torchtext_dataset_manager(FLAGS):
    which_part = [DATA_PART.Train, DATA_PART.Val, DATA_PART.Test]
    dm = DatasetManager(FLAGS, which_part)
    dl = dm.get_dataloader(DATA_PART.Train)
    for data in dl:
        packed_data = dm.pack_data(data, DATA_PART.Train)
