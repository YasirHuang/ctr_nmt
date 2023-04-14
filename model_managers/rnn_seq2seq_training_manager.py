# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:rnn_seq2seq_training_manager.py
@time:2020-08-28 
@desc:
'''

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from models.basic_models import RNN
from models.basic_models import EmbeddingLayer
from models.basic_models import LinearLayer

from models.rnn_seq2seq_models import RNNSeq2SeqEncoder
from models.rnn_seq2seq_models import RNNSeq2SeqDecoder
from models.rnn_seq2seq_models import RNNSeq2SeqAttention
from models.rnn_seq2seq_models import BahdanauAttentionDecoder, LuongAttentionDecoder
from models.rnn_seq2seq_models import RNNSeq2SeqModel
from models.rnn_seq2seq_models import RNNSeq2SeqGenerator
from models.utils import generate_mask

from model_managers.utils import beam_search_decode
from model_managers.basic_training_manager import BasicTrainingManager

from utils.config import MODE


class RNNSeq2SeqTrainingManager(BasicTrainingManager):
    def __init__(self, mode: MODE, FLAGS, log=True):
        super(RNNSeq2SeqTrainingManager, self).__init__(mode, FLAGS, log)

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

        # self.dropout = FLAGS.dropout #if mode != MODE.Infer else 0
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
        # data parameters
        self.src_vocab_size = FLAGS.src_vocab_size  # requires dataset to be prepared
        self.tgt_vocab_size = FLAGS.tgt_vocab_size  # requires dataset to be prepared
        self.batch_size = FLAGS.batch_size
        self.batch_first = FLAGS.batch_first
        # training parameters
        self.pretrained_embedding = FLAGS.pretrained_embedding  # specify in shell script
        # pretrained embedding
        self.source_embeddings_path = FLAGS.source_embeddings_path
        self.embedding_word_dict_path = FLAGS.embedding_word_dict_path
        # others
        self.pad_token = FLAGS.pad
        self.pad_token_id = FLAGS.pad_token_id

    def create_decoder(self,
                       recurrent_layer_type,
                       tgt_vocab_size,
                       embedding_dim,
                       encoder_output_dim,
                       attention_type,
                       attention_dim,
                       decoder_style,
                       decoder_dim,
                       decoder_num_layers,
                       attention_dropout,
                       decoder_dropout,
                       embedding_dropout,
                       batch_first,
                       bias=True,
                       project_out=True,
                       input_feeding=True
                       ):
        tgt_embedding_layer = EmbeddingLayer(tgt_vocab_size, embedding_dim, embedding_dropout)
        # tgt_embedding_layer = EmbeddingLayer(tgt_vocab_size, embedding_dim, dropout)
        decoder_input_dim = embedding_dim
        if decoder_style == 'bahdanau':
            decoder_input_dim += attention_dim if project_out else encoder_output_dim
        elif decoder_style == 'luong':
            if input_feeding:
                decoder_input_dim += attention_dim if project_out else (encoder_output_dim + decoder_dim)
        decoder_layer = RNNSeq2SeqDecoder(layer_type=recurrent_layer_type,
                                          layer_number=decoder_num_layers,
                                          input_dim=decoder_input_dim,
                                          output_dim=decoder_dim,
                                          dropout=decoder_dropout,
                                          batch_first=batch_first,
                                          bidirectional=False,
                                          bias=bias)
        attention = RNNSeq2SeqAttention(context_dim=encoder_output_dim,
                                        query_dim=decoder_dim,
                                        attention_dim=attention_dim,
                                        attention_type=attention_type,
                                        project_out=project_out,
                                        dropout=attention_dropout)
        if decoder_style == 'bahdanau':
            attention_decoder = BahdanauAttentionDecoder(attention,
                                                         decoder_layer)
        elif decoder_style == 'luong':
            attention_decoder = LuongAttentionDecoder(attention,
                                                      decoder_layer,
                                                      input_feeding=input_feeding)
        else:
            raise ValueError("Unknown decoder style: %s." % decoder_style)
        return tgt_embedding_layer, attention_decoder

    def create_model(self):
        src_embedding_layer = EmbeddingLayer(self.src_vocab_size,
                                             self.embedding_dim,
                                             self.embedding_dropout)
        # src_embedding_layer = EmbeddingLayer(self.src_vocab_size, self.embedding_dim, self.dropout)
        encoder = RNNSeq2SeqEncoder(layer_type=self.recurrent_layer_type,
                                    layer_number=self.encoder_num_layers,
                                    input_dim=self.embedding_dim,
                                    output_dim=self.encoder_dim,
                                    dropout=self.encoder_dropout,
                                    batch_first=self.batch_first,
                                    bidirectional=self.bidirectional,
                                    bias=self.bias)

        tgt_embedding_layer, decoder = self.create_decoder(
            recurrent_layer_type=self.recurrent_layer_type,
            tgt_vocab_size=self.tgt_vocab_size,
            embedding_dim=self.embedding_dim,
            encoder_output_dim=encoder.output_dim,
            attention_type=self.attention_type,
            attention_dim=self.attention_dim,
            decoder_style=self.decoder_style,
            decoder_dim=self.decoder_dim,
            decoder_num_layers=self.decoder_num_layers,
            attention_dropout=self.attention_dropout,
            decoder_dropout=self.decoder_dropout,
            embedding_dropout=self.embedding_dropout,
            batch_first=self.batch_first,
            bias=self.bias,
            project_out=self.project_out,
            input_feeding=self.input_feeding
        )
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

        generate_in_dim = decoder.output_dim
        # generator = nn.Sequential(
        #     nn.Linear(generate_in_dim, self.tgt_vocab_size),
        #     nn.LogSoftmax(dim=-1))
        generator = RNNSeq2SeqGenerator(LinearLayer(generate_in_dim,
                                                    self.tgt_vocab_size,
                                                    bias=self.generator_bias,
                                                    dropout=self.generator_dropout))
        seq2seq_model = RNNSeq2SeqModel(src_embedding_layer,
                                        encoder,
                                        decoder,
                                        tgt_embedding_layer,
                                        generator,
                                        encoder_initial_layer,
                                        decoder_initial_layer)

        self.model = seq2seq_model
        return self.model

    def freeze_parameters(self, model=None):
        if model is not None:
            assert model is self.model
        else:
            model = self.model
        if self.pretrained_embedding:
            for p in model.src_embedding.parameters():
                p.requires_grad = False

    def load_pretrained_model(self, model, dataset_manager):
        if self.pretrained_embedding:
            if self.multiple_gpu:
                model.module.src_embedding.load_embeddings_from_numpy(self.source_embeddings_path,
                                                                      self.embedding_word_dict_path,
                                                                      dataset_manager.src_word_to_id_dict,
                                                                      self.pad_token)
            else:
                model.src_embedding.load_embeddings_from_numpy(self.source_embeddings_path,
                                                               self.embedding_word_dict_path,
                                                               dataset_manager.src_word_to_id_dict,
                                                               self.pad_token)

    def loss_forward(self, epoch, i, packed_data, model, criterions=None, device=None):
        if not device:
            device = self.device

        if not criterions:
            criterions = self.criterions

        criterion = criterions[self.criterion]

        sos_src = packed_data[0].to(device)
        src = sos_src[:, 1:]
        eos_src = packed_data[1].to(device)
        sos_tgt = packed_data[2].to(device)
        eos_tgt = packed_data[3].to(device)
        src_len = packed_data[4].squeeze(1).to(device) - 1
        tgt_len = packed_data[5].squeeze(1).to(device)
        if self.initialize_encoder_state or \
                (self.initialize_decoder_state and not self.initialize_decoder_with_encoder):
            assert len(packed_data) >= 7
            global_feat = packed_data[6].to(device)
        else:
            global_feat = None

        if self.initialize_encoder_state:
            encoder_initial_state = global_feat
        else:
            encoder_initial_state = None
        if self.initialize_decoder_state:
            decoder_initial_state = global_feat
        else:
            decoder_initial_state = None

        predictions = model(input_sequence=src,
                            input_sequence_length=src_len,
                            target_sequence=sos_tgt,
                            target_sequence_length=tgt_len,
                            encoder_initial_state=encoder_initial_state,
                            decoder_initial_state=decoder_initial_state)
        # if i % 100 == 0:
        #     print(predictions)
        predictions = pack_padded_sequence(predictions, tgt_len, True, False)
        e_txts = pack_padded_sequence(eos_tgt, tgt_len, True, False)
        num_of_tokens = float(e_txts.data.shape[0])
        num_of_sents = float(tgt_len.shape[0])

        # loss = criterion(predictions.view(-1, predictions.size(-1)), eos_tgt.view(-1))
        loss = criterion(predictions.data, e_txts.data)
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
        src = sos_src[:, 1:]
        # eos_src = packed_data[1].to(device)
        # sos_tgt = packed_data[2].to(device)
        # eos_tgt = packed_data[3].to(device)
        src_len = packed_data[4].squeeze(1).to(device) - 1
        # tgt_len = packed_data[5].squeeze(1).to(device)
        if self.initialize_encoder_state or \
                (self.initialize_decoder_state and not self.initialize_decoder_with_encoder):
            assert len(packed_data) >= 7
            global_feat = packed_data[6].to(device)
        else:
            global_feat = None

        if self.initialize_encoder_state:
            encoder_initial_state = global_feat
        else:
            encoder_initial_state = None
        if self.initialize_decoder_state:
            decoder_initial_state = global_feat
        else:
            decoder_initial_state = None

        encoded_sequence, encoded_sequence_length, encoder_last_state = model.encode(
            src, src_len,
            encoder_initial_state=encoder_initial_state)

        # decoder_initial_state = None
        encoder_last_state = encoder_last_state.flatten_directions() \
            if encoder_last_state.bidirectional else encoder_last_state
        state_to_initialize_decoder = model.initialize_decoder(
            decoder_initial_state if decoder_initial_state else encoder_last_state)

        max_decode_step = min(max_decode_step, int(max(src_len) * max_decode_step_ratio))
        seqs, att_scores = beam_search_decode(model.attention_decoder,
                                              model.generator,
                                              (encoded_sequence,),
                                              (encoded_sequence_length,),
                                              state_to_initialize_decoder,
                                              model.tgt_embedding,
                                              tgt_word_to_id_dict,
                                              tgt_sos,
                                              eos,
                                              beam_width,
                                              max_decode_step)
        return seqs, att_scores
