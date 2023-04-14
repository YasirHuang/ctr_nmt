# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:tasks.py
@time:2022/3/6 
@desc:
'''
import six
import random
import collections
import torch
from enum import Enum
from torch.nn.utils.rnn import pack_padded_sequence


class MODEL_TRAINING_MODE(Enum):
    NMT = 0
    ENTITY_I2T = 1
    ENTITY_T2I = 2
    ENTITY_I2RT = 3
    ENTITY_A2T = 4
    ENTITY_A2I = 5


class TaskPool(object):

    def __init__(self, FLAGS):

        self.task_names = ['nmt', 't2i', 'i2t', 'i2rt', 'a2t', 'a2i']

        total = 0.0
        for n in self.task_names:
            name = n + "_train_ratio"
            value = 0.0
            if hasattr(FLAGS, name):
                value = float(getattr(FLAGS, n + "_train_ratio"))
            setattr(self, n + "_train_ratio", value)
            total += value

        for n in self.task_names:
            name = n + "_train_ratio"
            value = getattr(self, name) / total
            setattr(self, name, value)

    @property
    def valid_tasks(self):
        valid_ts = []
        for task_name in self.task_names:
            if self.get(task_name) > 0.0:
                valid_ts.append(task_name)
        return valid_ts

    def get(self, task_name):
        return getattr(self, task_name + "_train_ratio")

    def set(self, task_name, value):
        setattr(self, task_name + "_train_ratio", value)

    def random_choose(self):
        r = random.random()

        base = 0.0
        for n in self.task_names:
            name = n + "_train_ratio"
            value = getattr(self, name)
            base += value
            if r < base:
                if n == "nmt":
                    return MODEL_TRAINING_MODE.NMT
                return getattr(MODEL_TRAINING_MODE, "ENTITY_" + n.upper())


class TransformerTranslationTask(object):

    @staticmethod
    def train(transformer, criterion,
              sos_src, src_len, src_padding_mask,
              sos_tgt, eos_tgt, tgt_len, tgt_padding_mask,
              loss_normalization_fn=None):
        predictions = transformer(sos_src.transpose(0, 1),
                                  src_padding_mask.squeeze(-1),
                                  sos_tgt.transpose(0, 1),
                                  tgt_padding_mask.squeeze(-1))

        predictions = pack_padded_sequence(predictions.transpose(0, 1), tgt_len, True, False)
        labels = pack_padded_sequence(eos_tgt, tgt_len, True, False)

        num_of_tokens = float(labels.data.shape[0])
        num_of_sents = float(src_len.shape[0])
        loss = criterion(predictions.data, labels.data)
        if loss_normalization_fn is not None:
            loss = loss_normalization_fn(loss, num_of_sents, num_of_tokens)

        return {"loss": loss, "num_of_tokens": num_of_tokens, "num_of_sents": num_of_sents}


class EntityReconstructionTask(object):

    @staticmethod
    def train_i2t(encoder, criterion,
                  sos_src, src_len, src_padding_mask,
                  src_img, img_mask,
                  loss_normalization_fn=None,
                  regressional=False):
        select_mask = torch.ge(img_mask, 0.5).cuda()
        labels = torch.masked_select(sos_src, select_mask)
        if regressional:
            mm_memory = encoder(
                sos_src.transpose(0, 1),
                src_padding_mask.squeeze(-1),
                src_img.transpose(0, 1),
                img_mask.transpose(0, 1)
            )
            model_output = mm_memory
            labels = encoder.src_embedding_layer._embed(labels)
            # labels = torch.masked_select(
            #     label_embeddings, select_mask.unsqueeze(-1)).view(-1, label_embeddings.shape[-1])
        else:
            predictions = encoder(
                sos_src.transpose(0, 1),
                src_padding_mask.squeeze(-1),
                src_img.transpose(0, 1),
                img_mask.transpose(0, 1),
                require_logits=True
            )
            model_output = predictions
        num_of_tokens = float(len(labels))
        num_of_sents = float(src_len.shape[0])
        loss = criterion(model_output, labels)

        if loss_normalization_fn is not None:
            loss = loss_normalization_fn(loss, num_of_sents, num_of_tokens)
        return {"loss": loss, "num_of_tokens": num_of_tokens, "num_of_sents": num_of_sents}

    @staticmethod
    def train_t2i(encoder, criterion,
                  sos_src, src_len, src_padding_mask,
                  src_img, img_mask,
                  loss_normalization_fn=None,
                  project_image_before_loss=False,
                  project_to_image=False
                  ):
        selected_memory = encoder(
            src=sos_src.transpose(0, 1),
            src_padding_mask=src_padding_mask.squeeze(-1),
            img=None, img_mask=img_mask.transpose(0, 1), require_logits=False,
            project_to_image=project_to_image)

        select_mask = torch.ge(img_mask, 0.5).unsqueeze(-1).cuda()
        labels = torch.masked_select(src_img, select_mask).view(-1, src_img.size(-1))
        if project_image_before_loss:
            labels = encoder.project_image(labels)

        num_of_tokens = float(len(labels))
        num_of_sents = float(src_len.shape[0])
        loss = criterion(selected_memory, labels)
        if loss_normalization_fn is not None:
            loss = loss_normalization_fn(loss, num_of_sents, num_of_tokens)

        return {"loss": loss, "num_of_tokens": num_of_tokens, "num_of_sents": num_of_sents}

    @staticmethod
    def train_i2rt(encoder, criterion,
                   sos_src, src_len, src_padding_mask,
                   src_img, img_mask,
                   loss_normalization_fn=None,
                   fill_value=0,
                   mask_ratio=0.3,
                   regressional=False):
        image_mask = torch.ge(img_mask, 0.5).cuda()
        random_mask = torch.rand_like(img_mask).masked_fill_(image_mask, 1.0) < mask_ratio
        labels = torch.masked_select(sos_src, random_mask)

        masked_sos_src = torch.masked_fill(sos_src, random_mask, fill_value)

        if regressional:
            mm_memory = encoder(
                masked_sos_src.transpose(0, 1),
                src_padding_mask.squeeze(-1),
                src_img.transpose(0, 1),
                img_mask.transpose(0, 1),
                select_mask=random_mask.transpose(0, 1).unsqueeze(-1)
            )
            model_output = mm_memory
            labels = encoder.src_embedding_layer._embed(labels)
            # labels = torch.masked_select(
            #     label_embeddings, select_mask.unsqueeze(-1)).view(-1, label_embeddings.shape[-1])
        else:
            predictions = encoder(
                masked_sos_src.transpose(0, 1),
                src_padding_mask.squeeze(-1),
                src_img.transpose(0, 1),
                img_mask.transpose(0, 1),
                require_logits=True,
                select_mask=random_mask.transpose(0, 1).unsqueeze(-1)
            )
            model_output = predictions
        num_of_tokens = float(len(labels))
        num_of_sents = float(src_len.shape[0])
        loss = criterion(model_output, labels)
        if loss_normalization_fn is not None:
            loss = loss_normalization_fn(loss, num_of_sents, num_of_tokens)
        return {"loss": loss, "num_of_tokens": num_of_tokens, "num_of_sents": num_of_sents}

    @staticmethod
    def train_a2t(encoder, criterion,
                  sos_src, src_len, src_padding_mask,
                  loss_normalization_fn=None,
                  fill_value=0,
                  mask_ratio=0.3,
                  ):
        random_mask = torch.rand_like(sos_src) < mask_ratio
        masked_sos_src = torch.masked_fill(sos_src, random_mask, fill_value)
        labels = torch.masked_select(sos_src, random_mask)

        predictions = encoder(
            src=masked_sos_src.transpose(0, 1),
            src_padding_mask=src_padding_mask.squeeze(-1),
            img=None, img_mask=None, require_logits=True,
            select_mask=random_mask.transpose(0, 1).unsqueeze(-1)
        )

        num_of_tokens = float(len(labels))
        num_of_sents = float(src_len.shape[0])
        loss = criterion(predictions, labels)
        if loss_normalization_fn is not None:
            loss = loss_normalization_fn(loss, num_of_sents, num_of_tokens)

        return {"loss": loss, "num_of_tokens": num_of_tokens, "num_of_sents": num_of_sents}

    @staticmethod
    def train_a2i(encoder, criterion,
                  sos_src, src_len, src_padding_mask,
                  loss_normalization_fn=None,
                  fill_value=0,
                  mask_ratio=0.3,
                  ):
        return None
