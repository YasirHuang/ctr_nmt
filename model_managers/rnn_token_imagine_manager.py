# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:rnn_token_imagine_manager.py
@time:2020/11/18 
@desc:
'''
import random
import time
import sys

import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from models.basic_models import LinearLayer
from models.basic_models import EmbeddingLayer
from models.basic_models import build_project_layer
from models.rnn_seq2seq_models import RNNSeq2SeqEncoder
from models.rnn_seq2seq_models import RNNSeq2SeqDecoder
from models.rnn_seq2seq_models import RNNSeq2SeqAttention
from models.rnn_seq2seq_models import BahdanauAttentionDecoder
from models.rnn_seq2seq_models import LuongAttentionDecoder
from models.rnn_seq2seq_models import RNNSeq2SeqGenerator
from models.rnn_seq2seq_models import RNNSeq2SeqModel
from models.rnn_token_imagine_mmt import MultimodalEmbedding
from models.rnn_token_imagine_mmt import RNNTokenImagineModel
from models.rnn_token_imagine_mmt import DualModule
from models.utils import aeq, generate_mask

from model_managers.basic_training_manager import BasicTrainingManager
from model_managers.transformer_imagine_training_manager import MODEL_TRAINING_MODE
from model_managers.optimizer import DualOptimizer
from model_managers.schedular import DualScheduler
from model_managers.utils import beam_search_decode
from model_managers.utils import ppl as calculate_ppl

from utils.config import MODE
from utils.utils import safe_division

class RNNTokenImagineManager(BasicTrainingManager):
    def __init__(self, mode: MODE, FLAGS, log=True):
        super(RNNTokenImagineManager, self).__init__(mode, FLAGS, log)

        # create model parameters
        self.recurrent_layer_type = FLAGS.cell_type
        self.embedding_dim = FLAGS.text_embedding_size
        self.d_model = self.embedding_dim
        self.encoder_dim = FLAGS.encoder_num_units
        self.encoder_num_layers = FLAGS.encoder_num_layers
        self.attention_type = FLAGS.attention_type
        self.attention_dim = FLAGS.attention_num_units
        self.decoder_style = FLAGS.decoder_style
        self.decoder_dim = FLAGS.decoder_num_units
        self.decoder_num_layers = FLAGS.decoder_num_layers
        self.generator_bias = FLAGS.generator_bias

        self.embedding_dropout = FLAGS.embedding_dropout
        self.encoder_dropout = FLAGS.encoder_dropout
        self.decoder_dropout = FLAGS.decoder_dropout
        self.attention_dropout = FLAGS.attention_dropout
        self.generator_dropout = FLAGS.generator_dropout
        self.bidirectional = FLAGS.bidirectional
        self.bias = FLAGS.cell_bias
        self.project_out = FLAGS.project_out
        self.input_feeding = FLAGS.input_feeding
        self.initialize_encoder_state = FLAGS.initialize_encoder_state
        self.encoder_initial_state_size = FLAGS.encoder_initial_state_size
        self.initialize_decoder_state = FLAGS.initialize_decoder_state
        self.initialize_decoder_with_encoder = FLAGS.initialize_decoder_with_encoder
        self.decoder_initial_state_size = FLAGS.decoder_initial_state_size

        # image projector related parameters
        self.global_image_feature_size = FLAGS.global_image_feature_size
        # self.image_embedding_size = FLAGS.image_embedding_size
        self.big_image_projector = FLAGS.big_image_projector
        self.num_image_project_layer = FLAGS.num_image_project_layer
        self.image_projector_activation = FLAGS.image_projector_activation
        self.image_projector_bias = FLAGS.image_projector_bias
        self.image_projector_dropout = FLAGS.image_projector_dropout

        self.imagine_to_src = FLAGS.imagine_to_src
        self.share_decoder = FLAGS.share_decoder
        self.share_optimizer = FLAGS.share_optimizer
        self.share_embedding = FLAGS.share_embedding
        self.nmt_train_ratio = FLAGS.nmt_train_ratio

        # data parameters
        self.src_vocab_size = FLAGS.src_vocab_size  # requires dataset to be prepared
        self.tgt_vocab_size = FLAGS.tgt_vocab_size  # requires dataset to be prepared
        self.batch_size = FLAGS.batch_size
        self.batch_first = FLAGS.batch_first

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

    def create_embedding(self):
        image_projector = build_project_layer(
            num_layer=self.num_image_project_layer,
            input_dim=self.global_image_feature_size,
            output_dim=self.embedding_dim,
            activation=self.image_projector_activation,
            bias=self.image_projector_bias,
            dropout=self.image_projector_dropout,
            is_big=self.big_image_projector)
        if self.share_embedding:
            aeq(self.src_vocab_size, self.tgt_vocab_size)
            embedding = EmbeddingLayer(self.src_vocab_size, self.embedding_dim, self.embedding_dropout)
            multimodal_embedding = MultimodalEmbedding(embedding, image_projector)
            return multimodal_embedding, embedding, embedding
        src_embedding = EmbeddingLayer(self.src_vocab_size, self.embedding_dim, self.embedding_dropout)
        tgt_embedding = EmbeddingLayer(self.tgt_vocab_size, self.embedding_dim, self.embedding_dropout)
        multimodal_embedding = MultimodalEmbedding(src_embedding, image_projector)
        return multimodal_embedding, src_embedding, tgt_embedding

    def create_decoder(self):
        decoder_input_dim = self.embedding_dim
        encoder_output_dim = getattr(self, 'encoder_output_dim')
        if self.decoder_style == 'bahdanau':
            decoder_input_dim += self.attention_dim if self.project_out else encoder_output_dim
        elif self.decoder_style == 'luong':
            if self.input_feeding:
                decoder_input_dim += self.attention_dim if self.project_out else (encoder_output_dim + self.decoder_dim)
        decoder_layer = RNNSeq2SeqDecoder(layer_type=self.recurrent_layer_type,
                                          layer_number=self.decoder_num_layers,
                                          input_dim=decoder_input_dim,
                                          output_dim=self.decoder_dim,
                                          dropout=self.decoder_dropout,
                                          batch_first=self.batch_first,
                                          bidirectional=False,
                                          bias=self.bias)
        attention = RNNSeq2SeqAttention(context_dim=encoder_output_dim,
                                        query_dim=self.decoder_dim,
                                        attention_dim=self.attention_dim,
                                        attention_type=self.attention_type,
                                        project_out=self.project_out,
                                        dropout=self.attention_dropout)
        if self.decoder_style == 'bahdanau':
            attention_decoder = BahdanauAttentionDecoder(attention,
                                                         decoder_layer)
        elif self.decoder_style == 'luong':
            attention_decoder = LuongAttentionDecoder(attention,
                                                      decoder_layer,
                                                      input_feeding=self.input_feeding)
        else:
            raise ValueError("Unknown decoder style: %s." % self.decoder_style)
        return attention_decoder

    def create_model(self):
        mm_emb, src_emb, tgt_emb = self.create_embedding()
        encoder = RNNSeq2SeqEncoder(layer_type=self.recurrent_layer_type,
                                    layer_number=self.encoder_num_layers,
                                    input_dim=self.embedding_dim,
                                    output_dim=self.encoder_dim,
                                    dropout=self.encoder_dropout,
                                    batch_first=self.batch_first,
                                    bidirectional=self.bidirectional,
                                    bias=self.bias)
        setattr(self, "encoder_output_dim", encoder.output_dim)

        decoder = self.create_decoder()
        generate_in_dim = decoder.output_dim
        generator = RNNSeq2SeqGenerator(LinearLayer(generate_in_dim,
                                                    self.tgt_vocab_size,
                                                    bias=self.generator_bias,
                                                    dropout=self.generator_dropout))
        if self.share_decoder:
            mm_decoder = decoder
            mm_generator = generator
        else:
            mm_decoder = self.create_decoder()
            mm_generate_in_dim = mm_decoder.output_dim
            # TODO: error length should be self.src_vocab_size if self.imagine_to_src else self.tgt_vocab_size
            mm_generator = RNNSeq2SeqGenerator(LinearLayer(mm_generate_in_dim,
                                                           self.src_vocab_size if self.imagine_to_src else self.tgt_vocab_size,
                                                           bias=self.generator_bias,
                                                           dropout=self.generator_dropout))

        # if initialize encoder or not
        if self.initialize_encoder_state:
            encoder_initial_layer = LinearLayer(self.encoder_initial_state_size, self.encoder_dim)
        else:
            encoder_initial_layer = None

        # if initialize decoder or not
        # if initialize decoder with encoder last output state,
        # encoder directions should be considered
        if self.initialize_decoder_state:
            if self.initialize_decoder_with_encoder:
                if self.bidirectional:
                    decoder_initial_layer = LinearLayer(encoder.output_dim, decoder.output_dim)
                else:
                    decoder_initial_layer = lambda x: x
            else:
                decoder_initial_layer = LinearLayer(self.decoder_initial_state_size, decoder.output_dim)
        else:
            decoder_initial_layer = None

        seq2seq_model = RNNSeq2SeqModel(src_emb,
                                        encoder,
                                        decoder,
                                        tgt_emb,
                                        generator,
                                        encoder_initial_layer,
                                        decoder_initial_layer)
        imagine_model = RNNTokenImagineModel(
            mm_emb,
            encoder,
            mm_decoder,
            src_emb if self.imagine_to_src else tgt_emb,
            mm_generator,
            encoder_initial_layer=encoder_initial_layer,
            decoder_initial_layer=decoder_initial_layer)

        dual_model = DualModule(singlemodal_model=seq2seq_model, multimodal_model=imagine_model)
        self.model = dual_model
        return self.model

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
        encoder_initial_state = None
        decoder_initial_state = None

        if sos_src.shape[1] < src_with_eos.shape[1]:
            assert sos_src.shape[1] + 1 == src_with_eos.shape[1]
            original_sos_src = src_with_eos[:,:-1]
            eos_src = src_with_eos[:,1:]
        else:
            original_sos_src = None
            eos_src = src_with_eos

        if random.random() < self.nmt_train_ratio:
            self.current_mode = MODEL_TRAINING_MODE.NMT
            # if original_sos_src is not None:
            #     torch.set_printoptions(profile="full")
            #     print(sos_src)
            #     print(eos_src)
            #     print(original_sos_src)
            #     print(src_with_eos)
            #     return
            # for i in range(sos_src.shape[0]):
            #     for j in range(sos_src.shape[1] - 1):
            #         if sos_src[i][j+1] != eos_src[i][j]:
            #             sos_src[i][j+1] = torch.Tensor(eos_src[i][j]).int()
            #             torch.set_printoptions(profile="full")
            #             print(sos_src)
            #             print(eos_src)
            #             assert sos_src[i][j+1] != eos_src[i][j], "%d(%d),%d(%d)" %(sos_src[i][j+1],i,eos_src[i][j],j)
            predictions = model.singlemodal(sos_src if original_sos_src is None else original_sos_src,
                                            src_len,
                                            sos_tgt,
                                            tgt_len,
                                            encoder_initial_state=encoder_initial_state,
                                            decoder_initial_state=decoder_initial_state)

            predictions = pack_padded_sequence(predictions, tgt_len, True, False)
            labels = pack_padded_sequence(eos_tgt, tgt_len, True, False)

            if not self.share_optimizer:
                self.optimizer.set_current_optimizer(self.optimizer.singlemodal_optimizer)
        else:
            self.current_mode = MODEL_TRAINING_MODE.IMAGINATE
            if self.imagine_to_src:
                the_sos_tgt = sos_src
                the_eos_tgt = eos_src
                the_tgt_len = src_len
            else:
                the_sos_tgt = sos_tgt
                the_eos_tgt = eos_tgt
                the_tgt_len = tgt_len
            predictions = model.multimodal(sos_src,
                                           src_len,
                                           src_img,
                                           img_mask,
                                           the_sos_tgt,  # notice the difference here
                                           the_tgt_len,  # notice the difference here
                                           encoder_initial_state=encoder_initial_state,
                                           decoder_initial_state=decoder_initial_state)
            predictions = pack_padded_sequence(predictions, the_tgt_len, True, False)
            labels = pack_padded_sequence(the_eos_tgt, the_tgt_len, True, False)

            if not self.share_optimizer:
                self.optimizer.set_current_optimizer(self.optimizer.multimodal_optimizer)
        # if not self.share_decoder:
        #     self.optimizer.set_current_mode(self.current_mode)

        # norm = float(labels.data.shape[0])
        num_of_tokens = float(labels.data.shape[0])
        num_of_sents = float(tgt_len.shape[0])
        loss = criterion(predictions.data, labels.data)
        loss = self.normalize_loss(loss, num_of_sents, num_of_tokens)
        loss = loss / self.optimize_delay
        batch_dict = {'loss': loss,
                      "num_of_sents": num_of_sents, "num_of_tokens": num_of_tokens}
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

        encoder_initial_state = None
        decoder_initial_state = None
        nmt_model = model.singlemodal
        encoded_sequence, encoded_sequence_length, encoder_last_state = nmt_model.encode(
            sos_src if original_sos_src is None else original_sos_src, src_len,
            encoder_initial_state=encoder_initial_state)

        # decoder_initial_state = None
        encoder_last_state = encoder_last_state.flatten_directions() \
            if encoder_last_state.bidirectional else encoder_last_state
        state_to_initialize_decoder = nmt_model.initialize_decoder(
            decoder_initial_state if decoder_initial_state else encoder_last_state)

        max_decode_step = min(max_decode_step, int(max(src_len) * max_decode_step_ratio))
        seqs, att_scores = beam_search_decode(nmt_model.attention_decoder,
                                              nmt_model.generator,
                                              (encoded_sequence,),
                                              (encoded_sequence_length,),
                                              state_to_initialize_decoder,
                                              nmt_model.tgt_embedding,
                                              tgt_word_to_id_dict,
                                              tgt_sos,
                                              eos,
                                              beam_width,
                                              max_decode_step)

        return seqs, att_scores


    def imaginate_hypo_forward(self,
                     packed_data,
                     model,
                     tgt_word_to_id_dict,
                     tgt_vocab_size,
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

        encoder_initial_state = None
        decoder_initial_state = None
        mmt_model = model.multimodal

        encoded_sequence, encoded_sequence_length, encoder_last_state = mmt_model.encode(
            sos_src, src_len,
            src_img, img_mask,
            encoder_initial_state=encoder_initial_state)

        # decoder_initial_state = None
        encoder_last_state = encoder_last_state.flatten_directions() \
            if encoder_last_state.bidirectional else encoder_last_state
        state_to_initialize_decoder = mmt_model.initialize_decoder(
            decoder_initial_state if decoder_initial_state else encoder_last_state)

        max_decode_step = min(max_decode_step, int(max(src_len) * max_decode_step_ratio))
        seqs, att_scores = beam_search_decode(mmt_model.attention_decoder,
                                              mmt_model.generator,
                                              (encoded_sequence,),
                                              (encoded_sequence_length,),
                                              state_to_initialize_decoder,
                                              mmt_model.tgt_embedding,
                                              tgt_word_to_id_dict,
                                              tgt_sos,
                                              eos,
                                              beam_width,
                                              max_decode_step,
                                              tgt_vocab_size=tgt_vocab_size)

        return seqs, att_scores

    def create_optimizer(self, optimizer_name, learning_rate, model):
        if self.share_optimizer:
            return super(RNNTokenImagineManager, self).create_optimizer(
                optimizer_name, learning_rate, model)

        else:
            nmt_optim = super(RNNTokenImagineManager, self).create_optimizer(
                optimizer_name, learning_rate, model.singlemodal_model)
            img_optim = super(RNNTokenImagineManager, self).create_optimizer(
                optimizer_name, learning_rate, model.multimodal_model)
            self.optimizer = DualOptimizer(nmt_optim, img_optim)

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

        mm_tls = safe_division(self.training_imaginate_running_loss, self.training_imaginate_num_of_tokens)
        mm_sls = safe_division(self.training_imaginate_running_loss, self.training_imaginate_num_of_sents)
        sm_tls = safe_division(self.training_running_loss, self.training_num_of_tokens)
        sm_sls = safe_division(self.training_running_loss, self.training_num_of_sents)

        # mm_tls = self.training_imaginate_running_loss / self.training_imaginate_num_of_tokens
        # mm_sls = self.training_imaginate_running_loss / self.training_imaginate_num_of_sents
        # sm_tls = self.training_running_loss / self.training_num_of_tokens
        # sm_sls = self.training_running_loss / self.training_num_of_sents
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
                  safe_division((time.time() - self.training_start_time), self.steps_per_internal_eval),
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
            return (mm_tls + sm_tls) / 2.0
        else:
            return (mm_sls + sm_sls) / 2.0

    def get_learning_rate(self):
        if self.share_optimizer:
            return super(RNNTokenImagineManager, self).get_learning_rate()
        else:
            return self.optimizer.singlemodal_optimizer.state_dict()['param_groups'][0]['lr'], \
                   self.optimizer.multimodal_optimizer.state_dict()['param_groups'][0]['lr']

    def set_scheduler(self, optimizer, step_size, gamma):
        if not self.share_optimizer:
            if self.scheduler_type == "StepLR":
                scheduler0 = StepLR(optimizer.singlemodal_optimizer, step_size=step_size, gamma=gamma)
                scheduler1 = StepLR(optimizer.multimodal_optimizer, step_size=step_size, gamma=gamma)
            elif self.scheduler_type == "ExponentialLR":
                scheduler0 = ExponentialLR(optimizer.singlemodal_optimizer, gamma=gamma)
                scheduler1 = ExponentialLR(optimizer.multimodal_optimizer, gamma=gamma)
            else:
                raise ValueError("Unsupported scheduler type %s." % self.scheduler_type)
            self.scheduler = DualScheduler(scheduler0, scheduler1)
            return self.scheduler
        else:
            return super(RNNTokenImagineManager, self).set_scheduler(optimizer, step_size, gamma)

