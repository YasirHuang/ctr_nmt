# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:transformer_imagine_training_manager.py
@time:2020-09-20 
@desc:
'''
import os
import random
import time
import sys
from enum import Enum

import torch
import torch.nn as nn

import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import LayerNorm
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer

from models.basic_models import EmbeddingLayer, LinearLayer
from models.basic_models import build_project_layer
from models.transformer import PositionalEncoder
from models.transformer import PositionalEmbedding
from models.transformer import TransformerModel
from models.transformer import Generator
from models.transformer_imagine import MultimodalEmbedding
from models.transformer_imagine import TransformerImagineModel
from models.transformer_imagine import DualModule
from models.utils import generate_mask

from model_managers.utils import beam_search_decode_transformer
from model_managers.basic_training_manager import BasicTrainingManager
from model_managers.optimizer import NoamOpt, DualNoamOpt
from model_managers.utils import ppl as calculate_ppl

from utils.config import MODE
from utils import early_stop
from utils.utils import safe_division as sd


class MODEL_TRAINING_MODE(Enum):
    NMT = 0
    IMAGINATE = 1


class TransformerImagineTrainingManager(BasicTrainingManager):
    def __init__(self, mode: MODE, FLAGS, log=True):
        super(TransformerImagineTrainingManager, self).__init__(mode, FLAGS, log)

        # create model parameters
        # self.share_embedding = FLAGS.share_embedding
        self.share_decoder = FLAGS.share_decoder
        self.number_of_head = FLAGS.number_of_head
        self.embedding_dim = FLAGS.text_embedding_size
        self.d_model = self.embedding_dim
        self.encoder_num_layers = FLAGS.encoder_num_layers
        self.encoder_dim = FLAGS.encoder_num_units
        self.decoder_num_layers = FLAGS.decoder_num_layers
        self.decoder_dim = FLAGS.decoder_num_units
        self.activation = FLAGS.transformer_activation
        self.generator_bias = FLAGS.generator_bias

        self.dropout = FLAGS.dropout
        self.embedding_dropout = FLAGS.embedding_dropout
        self.encoder_dropout = FLAGS.encoder_dropout
        self.decoder_dropout = FLAGS.decoder_dropout
        self.generator_dropout = FLAGS.generator_dropout
        self.image_projector_dropout = FLAGS.image_projector_dropout

        # image projector related parameters
        self.global_image_feature_size = FLAGS.global_image_feature_size
        # self.image_embedding_size = FLAGS.image_embedding_size
        self.big_image_projector = FLAGS.big_image_projector
        self.num_image_project_layer = FLAGS.num_image_project_layer
        self.image_projector_activation = FLAGS.image_projector_activation
        self.image_projector_bias = FLAGS.image_projector_bias
        self.image_projector_dropout = FLAGS.image_projector_dropout

        # data parameters
        self.max_len = FLAGS.max_len
        self.src_vocab_size = FLAGS.src_vocab_size  # requires dataset to be prepared
        self.tgt_vocab_size = FLAGS.tgt_vocab_size  # requires dataset to be prepared
        self.batch_size = FLAGS.batch_size
        self.batch_first = FLAGS.batch_first
        # training parameters
        self.imagine_to_src = FLAGS.imagine_to_src
        self.share_optimizer =  FLAGS.share_optimizer
        self.nmt_train_ratio = FLAGS.nmt_train_ratio
        # self.start_decay_step = FLAGS.start_decay_step # already in basic_manager
        # self.optimize_delay = FLAGS.optimize_delay # already in basic_manager
        self.pretrained_embedding = FLAGS.pretrained_embedding  # specify in shell script
        self.multiple_gpu = FLAGS.multiple_gpu
        self.multitask_warmup = FLAGS.multitask_warmup

        # pretrained embedding
        self.source_embeddings_path = FLAGS.source_embeddings_path
        self.embedding_word_dict_path = FLAGS.embedding_word_dict_path
        # others
        self.imaginate_inference = FLAGS.imaginate_inference
        self.pad_token = FLAGS.pad
        self.pad_token_id = FLAGS.pad_token_id

        self.training_imaginate_running_loss = 0.0
        self.training_num_of_sents = 0
        self.training_imaginate_num_of_sents = 0
        self.training_num_of_tokens = 0
        self.training_imaginate_num_of_tokens = 0
        self.training_nmt_step = 0
        self.training_imaginate_step = 0
        self.nmt_steps_per_internal_eval = 0
        self.img_steps_per_internal_eval = 0
        self.current_mode = MODEL_TRAINING_MODE.NMT

    def create_model(self):
        # only support shared embeddings

        # create embeddings
        # create shared embedding
        shared_embedding_layer = EmbeddingLayer(self.src_vocab_size, self.embedding_dim)
        positional_encoder = PositionalEncoder(model_dim=self.embedding_dim,
                                               dropout=self.embedding_dropout,
                                               max_len=self.max_len * 2)

        image_projector = build_project_layer(
            num_layer=self.num_image_project_layer,
            input_dim=self.global_image_feature_size,
            output_dim=self.embedding_dim,
            activation=self.image_projector_activation,
            bias=self.image_projector_bias,
            dropout=self.image_projector_dropout,
            is_big=self.big_image_projector)
        # project_layer = LinearLayer(self.image_feature_dim, self.embedding_dim, dropout=self.image_projector_dropout)
        src_positional_mmembedding = MultimodalEmbedding(shared_embedding_layer,
                                                         image_projector,
                                                         positional_encoder)
        shared_positional_embedding = PositionalEmbedding(shared_embedding_layer, positional_encoder)

        # create encoder
        encoder_layer = TransformerEncoderLayer(d_model=self.embedding_dim,
                                                nhead=self.number_of_head,
                                                dim_feedforward=self.encoder_dim,
                                                dropout=self.encoder_dropout,
                                                activation=self.activation)
        layer_norm = LayerNorm(self.embedding_dim)
        encoder = TransformerEncoder(encoder_layer, self.encoder_num_layers, layer_norm)

        # create to-src decoder
        decoder_layer = TransformerDecoderLayer(d_model=self.embedding_dim,
                                                nhead=self.number_of_head,
                                                dim_feedforward=self.decoder_dim,
                                                dropout=self.decoder_dropout,
                                                activation=self.activation)
        layer_norm = LayerNorm(self.embedding_dim)
        decoder_for_multimodal = TransformerDecoder(decoder_layer, self.decoder_num_layers, layer_norm)
        # project_layer = LinearLayer(self.embedding_dim,
        #                             self.tgt_vocab_size,
        #                             bias=False)
        # project_layer.linear_layer.weight = shared_positional_embedding.embedding_layer.embedding.weight

        generator = Generator(LinearLayer(self.embedding_dim,
                                          self.tgt_vocab_size,
                                          dropout=self.generator_dropout,
                                          bias=self.generator_bias))
        generator.output_layer.linear_layer.weight = shared_positional_embedding.embedding_layer.embedding.weight
        # create to-tgt decoder
        if self.share_decoder:
            decoder_for_singlemodal = decoder_for_multimodal
        else:
            decoder_layer = TransformerDecoderLayer(d_model=self.embedding_dim,
                                                    nhead=self.number_of_head,
                                                    dim_feedforward=self.decoder_dim,
                                                    dropout=self.decoder_dropout,
                                                    activation=self.activation)
            layer_norm = LayerNorm(self.embedding_dim)
            decoder_for_singlemodal = TransformerDecoder(decoder_layer, self.decoder_num_layers, layer_norm)

        transformer = TransformerModel(shared_positional_embedding,
                                       encoder,
                                       decoder_for_singlemodal,
                                       shared_positional_embedding,
                                       generator)
        mm_transformer = TransformerImagineModel(src_positional_mmembedding,
                                                 encoder,
                                                 decoder_for_multimodal,
                                                 shared_positional_embedding,
                                                 generator)
        dual_module = DualModule(transformer, mm_transformer)

        self.model = dual_module
        return self.model

    def freeze_parameters(self, model=None):
        if model is not None:
            assert model is self.model
        else:
            model = self.model
        if self.pretrained_embedding:
            for p in model.src_embedding_layer.parameters():
                p.requires_grad = False

    def load_pretrained_model(self, model, dataset_manager):
        if self.pretrained_embedding:
            if self.multiple_gpu:
                model.module.src_embedding_layer.embedding_layer. \
                    load_embeddings_from_numpy(self.source_embeddings_path,
                                               self.embedding_word_dict_path,
                                               dataset_manager.src_word_to_id_dict,
                                               self.pad_token)
            else:
                model.src_embedding_layer.embedding_layer. \
                    load_embeddings_from_numpy(self.source_embeddings_path,
                                               self.embedding_word_dict_path,
                                               dataset_manager.src_word_to_id_dict,
                                               self.pad_token)

    def forward_hook(self, module, input, output):
        if not hasattr(self, "attenion_storages"):
            setattr(self, "attenion_storages", list())
        attns = getattr(self, "attenion_storages")
        cur_attn = output[1].detach().cpu()
        # beam_size, batch_size, target_sequence_length, source_sequence_length
        last_bsz, last_tsl, last_ssl = attns[-1].shape if len(attns) > 0 else (0, 0, 0)
        cur_bsz, cur_tsl, cur_ssl = cur_attn.shape
        if (
            (last_bsz == cur_bsz) and
            (last_tsl == cur_tsl - 1) and
            (last_ssl == cur_ssl)
        ):
            attns[-1] = cur_attn
        else:
            attns.append(cur_attn)

    def remove_hooks(self, *args, **kwargs):
        attns = getattr(self, "attenion_storages")

        attns_to_store = list()
        if "beam_width" in kwargs:
            beam_width = kwargs["beam_width"]
            print("beam_width: ", beam_width)
            for a in attns:
                attns_to_store.extend(
                    a.view(-1, beam_width, a.shape[1], a.shape[2])[:,0,:,:])
        else:
            for a in attns:
                attns_to_store.extend(a)

        if "sentences" in kwargs:
            print("fix attention scores by sentences")
            sentences = kwargs["sentences"]
            assert len(sentences) == len(attns_to_store)
            for i, (sent, attn) in enumerate(zip(sentences, attns_to_store)):
                tgt_sent_len = len(sent.split())
                src_sent_len = len(torch.masked_select(attn[0,:], attn[0,:]>0.000001))
                if src_sent_len < len(attn[0,:]) and attn[0,src_sent_len] > 0.0000001:
                    print("allert! attn:", attn, "tgt_len:", tgt_sent_len, "src_len:", src_sent_len)
                attns_to_store[i] = attn[:tgt_sent_len, :src_sent_len]

        out_dir = kwargs["out_dir"] if "out_dir" in kwargs else "."
        torch.save(attns_to_store, os.path.join(out_dir, "attn_scores.pth"))

        print("attention stored. total_size:", len(attns_to_store))

    def add_hooks(self):
        print("add_hooks")
        last_decoder_layer = self.model.transformer.transformer_decoder.layers[-1]
        for name, module in last_decoder_layer.named_modules():
            print(name)
            if name == "multihead_attn":
                module.register_forward_hook(self.forward_hook)
        sys.stdout.flush()


    def loss_forward(self, current_epoch, i, packed_data, model, criterions=None, device=None):
        # def switch_mode()
        if not device:
            device = self.device

        if not criterions:
            criterions = self.criterions

        criterion = criterions[self.criterion]

        sos_src = packed_data[0].to(device)
        src_with_eos = packed_data[1].to(device)
        sos_tgt = packed_data[2].to(device)
        eos_tgt = packed_data[3].to(device)
        src_len = packed_data[4].squeeze(1).to(device)
        tgt_len = packed_data[5].squeeze(1).to(device)
        src_img = packed_data[6].to(device)
        img_mask = packed_data[7].to(device)
        src_padding_mask = generate_mask(src_len, device=device)
        tgt_padding_mask = generate_mask(tgt_len, device=device)

        if sos_src.shape[1] < src_with_eos.shape[1]:
            assert sos_src.shape[1] + 1 == src_with_eos.shape[1]
            original_sos_src = src_with_eos[:,:-1]
            eos_src = src_with_eos[:,1:]
        else:
            original_sos_src = None
            eos_src = src_with_eos

        if self.training_total_step < self.multitask_warmup or \
                random.random() < self.nmt_train_ratio:
            self.current_mode = MODEL_TRAINING_MODE.NMT
            src_input = sos_src if original_sos_src is None else original_sos_src
            predictions = model.singlemodal(src_input.transpose(0, 1),
                                            src_padding_mask.squeeze(-1),
                                            sos_tgt.transpose(0, 1),
                                            tgt_padding_mask.squeeze(-1))

            predictions = pack_padded_sequence(predictions.transpose(0, 1), tgt_len, True, False)
            labels = pack_padded_sequence(eos_tgt, tgt_len, True, False)

            if not self.share_optimizer:
                self.optimizer.set_current_optimizer(self.optimizer.transformer_noamopt)
        else:
            self.current_mode = MODEL_TRAINING_MODE.IMAGINATE
            if self.imagine_to_src:
                the_sos_tgt = sos_src
                the_eos_tgt = eos_src
                the_tgt_len = src_len
                the_tgt_padding_mask = src_padding_mask
            else:
                the_sos_tgt = sos_tgt
                the_eos_tgt = eos_tgt
                the_tgt_len = tgt_len
                the_tgt_padding_mask = tgt_padding_mask
            predictions = model.multimodal(sos_src.transpose(0, 1),
                                           src_padding_mask.squeeze(-1),
                                           the_sos_tgt.transpose(0, 1),  # notice the difference here
                                           the_tgt_padding_mask.squeeze(-1),  # notice the difference here
                                           src_img.transpose(0, 1),
                                           img_mask.transpose(0, 1))
            predictions = pack_padded_sequence(predictions.transpose(0, 1), the_tgt_len, True, False)
            labels = pack_padded_sequence(the_eos_tgt, the_tgt_len, True, False)

            if not self.share_optimizer:
                self.optimizer.set_current_optimizer(self.optimizer.mmtransformer_noamopt)
        # if not self.share_decoder:
        #     self.optimizer.set_current_mode(self.current_mode)

        # norm = float(labels.data.shape[0])
        num_of_tokens = float(labels.data.shape[0])
        num_of_sents = float(tgt_len.shape[0])
        loss = criterion(predictions.data, labels.data)
        loss = self.normalize_loss(loss, num_of_sents, num_of_tokens)
        loss = loss / self.optimize_delay
        batch_dict = {'loss':loss ,
                      "num_of_sents":num_of_sents, "num_of_tokens":num_of_tokens}
        return batch_dict

    def hypo_forward(self,
                     packed_data,
                     model,
                     tgt_word_to_id_dict,
                     device=None,
                     beam_width=3,
                     max_decode_step=50,
                     max_decode_step_ratio=2.0,
                     src_sos='<s>',
                     tgt_sos='<s>',
                     eos='</s>',
                     unk='<unk'):
        if not device:
            device = self.device

        sos_src = packed_data[0].to(device)
        src_with_eos = packed_data[1].to(device)
        # sos_tgt = packed_data[2].to(device)
        # eos_tgt = packed_data[3].to(device)
        src_len = packed_data[4].squeeze(1).to(device)
        # tgt_len = packed_data[5].squeeze(1).to(device)

        if sos_src.shape[1] < src_with_eos.shape[1]:
            assert sos_src.shape[1] + 1 == src_with_eos.shape[1]
            original_sos_src = src_with_eos[:,:-1]
            eos_src = src_with_eos[:,1:]
        else:
            original_sos_src = None
            eos_src = src_with_eos

        src_input = sos_src if original_sos_src is None else original_sos_src

        src_padding_mask = generate_mask(src_len, device=device).squeeze(-1)
        transformer = model.transformer
        memory = transformer.encode(src_input.transpose(0, 1), src_padding_mask)

        max_decode_step = min(max_decode_step, int(max(src_len) * max_decode_step_ratio))
        seqs = beam_search_decode_transformer(transformer,
                                              memory,
                                              src_padding_mask,
                                              tgt_word_to_id_dict,
                                              sos=tgt_sos,
                                              eos=eos,
                                              beam_width=beam_width,
                                              max_decode_step=max_decode_step)

        return seqs, None


    def imaginate_hypo_forward(self,
                     packed_data,
                     model,
                     word_to_id_dict,
                     device=None,
                     beam_width=3,
                     max_decode_step=50,
                     max_decode_step_ratio=2.0,
                     src_sos='<s>',
                     tgt_sos='<s>',
                     eos='</s>',
                     unk='<unk'):
        if not device:
            device = self.device

        sos_src = packed_data[0].to(device)
        src_with_eos = packed_data[1].to(device)
        sos_tgt = packed_data[2].to(device)
        eos_tgt = packed_data[3].to(device)
        src_len = packed_data[4].squeeze(1).to(device)
        tgt_len = packed_data[5].squeeze(1).to(device)
        src_img = packed_data[6].to(device)
        img_mask = packed_data[7].to(device)
        src_padding_mask = generate_mask(src_len, device=device).squeeze(-1)
        tgt_padding_mask = generate_mask(tgt_len, device=device).squeeze(-1)


        transformer = model.multimodal_transformer
        memory = transformer.encode(sos_src.transpose(0, 1),
                                    src_padding_mask.squeeze(-1),
                                    src_img.transpose(0, 1),
                                    img_mask.transpose(0, 1))

        max_decode_step = min(max_decode_step, int(max(src_len) * max_decode_step_ratio))
        seqs = beam_search_decode_transformer(transformer,
                                              memory,
                                              src_padding_mask,
                                              word_to_id_dict,
                                              sos=src_sos if self.imagine_to_src else tgt_sos,
                                              eos=eos,
                                              beam_width=beam_width,
                                              max_decode_step=max_decode_step)

        return seqs, None

    def create_optimizer(self, optimizer_name, learning_rate, model):
        if self.share_optimizer:
            adamopt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 betas=(0.9, 0.98), eps=1e-9)
            self.optimizer = NoamOpt(self.embedding_dim,
                                     learning_rate,
                                     self.start_decay_step,
                                     adamopt,
                                     delay_update=self.optimize_delay)
        else:
            transformer_adam = optim.Adam(
                filter(lambda p: p.requires_grad, model.transformer.parameters()),
                betas=(0.9, 0.98), eps=1e-9)
            mmtransformer_adam = optim.Adam(
                filter(lambda p: p.requires_grad, model.multimodal_transformer.parameters()),
                betas=(0.9, 0.98), eps=1e-9)
            transformer_optimizer = NoamOpt(self.embedding_dim,
                                            learning_rate,
                                            self.start_decay_step,
                                            transformer_adam,
                                            delay_update=self.optimize_delay)
            mmtransformer_optimizer = NoamOpt(self.embedding_dim,
                                              learning_rate,
                                              self.start_decay_step,
                                              mmtransformer_adam,
                                              delay_update=self.optimize_delay)
            self.optimizer = DualNoamOpt(transformer_optimizer, mmtransformer_optimizer)

        return self.optimizer

    def training_one_more_step(self, batch_info):
        loss = batch_info['loss'].data.clone()
        num_of_tokens = batch_info['num_of_tokens']
        num_of_sents = batch_info['num_of_sents']
        step_loss = self.denormalize_loss(loss, num_of_sents, num_of_tokens)
        if self.current_mode == MODEL_TRAINING_MODE.NMT:
            # temporary
            self.training_running_loss += step_loss
            self.training_num_of_sents += num_of_sents
            self.training_num_of_tokens += num_of_tokens
            self.nmt_steps_per_internal_eval += 1

            # last for entire training
            self.training_nmt_step += 1
        else:
            # temporary
            self.training_imaginate_running_loss += step_loss
            self.training_imaginate_num_of_sents += num_of_sents
            self.training_imaginate_num_of_tokens += num_of_tokens
            self.img_steps_per_internal_eval += 1

            # last for entire training
            self.training_imaginate_step += 1

        self.training_total_step += 1
        self.steps_per_internal_eval += 1

    def reset_internal_evalation(self):
        self.training_running_loss = 0.0
        self.training_imaginate_running_loss = 0.0
        self.training_num_of_sents = 0
        self.training_imaginate_num_of_sents = 0
        self.training_num_of_tokens = 0
        self.training_imaginate_num_of_tokens = 0
        self.steps_per_internal_eval = 0
        self.img_steps_per_internal_eval = 0
        self.nmt_steps_per_internal_eval = 0

    def report_internal_evaluation(self,
                                   model,
                                   current_epoch,
                                   steps_in_epoch,
                                   summary_writer=None):
        # if self.share_optimizer:
        #     return super(TransformerImagineTrainingManager, self). \
        #         report_internal_evaluation(model,
        #                                    current_epoch,
        #                                    steps_in_epoch,
        #                                    summary_writer)
        mm_tls = sd(self.training_imaginate_running_loss, self.training_imaginate_num_of_tokens)
        mm_sls = sd(self.training_imaginate_running_loss, self.training_imaginate_num_of_sents)
        sm_tls = sd(self.training_running_loss, self.training_num_of_tokens)
        sm_sls = sd(self.training_running_loss, self.training_num_of_sents)
        mm_ppl = calculate_ppl(self.training_imaginate_running_loss, self.training_imaginate_num_of_tokens)
        sm_ppl = calculate_ppl(self.training_running_loss, self.training_num_of_tokens)
        lr = self.get_learning_rate()
        if self.share_optimizer:
            nmt_lr = lr
            mmt_lr = lr
        else:
            nmt_lr, mmt_lr = lr
        print("training info: epoch %d (%d/%d), "
              "time per step: %.2fs, total time: %.2fs" % (
                  current_epoch,
                  steps_in_epoch,
                  self.training_total_step,
                  (time.time() - self.training_start_time) / self.steps_per_internal_eval,
                  time.time() - self.training_start_time))
        print("\tnmt(%3d): sloss %.2f, tloss %.2f, ppl %.2f, lr %.1e" %
              (self.nmt_steps_per_internal_eval, sm_sls, sm_tls, sm_ppl, nmt_lr))
        print("\timg(%3d): sloss %.2f, tloss %.2f, ppl %.2f, lr %.1e" %
              (self.img_steps_per_internal_eval, mm_sls, mm_tls, mm_ppl, mmt_lr))
        self.reset_internal_evalation()
        self.now_as_start_time(MODE.Train)
        summary_writer.add_scalar(tag="train_single_modality_token_cross_entropy",
                                  scalar_value=sm_tls,
                                  global_step=self.training_total_step)
        summary_writer.add_scalar(tag="train_single_modality_sent_cross_entropy",
                                  scalar_value=sm_sls,
                                  global_step=self.training_total_step)
        summary_writer.add_scalar(tag="train_single_modality_ppl",
                                  scalar_value=sm_ppl,
                                  global_step=self.training_total_step)
        summary_writer.add_scalar(tag="train_multi_modality_token_cross_entropy",
                                  scalar_value=mm_tls,
                                  global_step=self.training_total_step)
        summary_writer.add_scalar(tag="train_multi_modality_sent_cross_entropy",
                                  scalar_value=mm_sls,
                                  global_step=self.training_total_step)
        summary_writer.add_scalar(tag="train_multi_modality_ppl",
                                  scalar_value=mm_ppl,
                                  global_step=self.training_total_step)
        sys.stdout.flush()
        if self.loss_normalize_type == 'token':
            return (mm_tls+sm_tls)/2.0
        else:
            return (mm_sls+sm_sls)/2.0

    def get_learning_rate(self):
        if self.share_optimizer:
            assert isinstance(self.optimizer, NoamOpt)
            return self.optimizer._rate
        else:
            return self.optimizer.transformer_noamopt._rate, \
                   self.optimizer.mmtransformer_noamopt._rate

