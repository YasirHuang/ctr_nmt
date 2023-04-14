# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:basic_models.py
@time:2020-08-27 
@desc:
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from models.basic_models import GlobalAttention, RNN, RNNCellState
from models.utils import train_decoder, generate_mask


class RNNSeq2SeqAttention(GlobalAttention):

    def forward(self,
                encoded_sequence,
                encoded_sequence_length,
                state_to_attend):
        mask = generate_mask(encoded_sequence_length)  # (bs, src_seq_len, tgt_seq_len=1)
        c, scores = super(RNNSeq2SeqAttention, self).forward(encoded_sequence,
                                                             mask,
                                                             state_to_attend)
        return c, scores


class RNNSeq2SeqDecoder(RNN):
    pass


class RNNSeq2SeqEncoder(RNN):
    pass


class RNNSeq2SeqAttentionDecoder(nn.Module):
    def __init__(self,
                 attention,
                 decoder,
                 batch_first=True):
        super(RNNSeq2SeqAttentionDecoder, self).__init__()
        self.attention = attention
        self.decoder = decoder

        self.batch_first = batch_first

    @property
    def output_dim(self):
        return NotImplementedError

    def attention_score_register(self):
        return None

    def register_attention_score(self, attention_score, attention_score_register, sorted_id):
        if attention_score_register is None:
            attention_score_register = attention_score[sorted_id].unsqueeze(1)
        else:
            attention_score_register = torch.cat(
                [attention_score_register[sorted_id], attention_score[sorted_id].unsqueeze(1)], dim=1)
        return attention_score_register

    def forward(self,
                encoded_sequence,
                encoded_sequence_length,
                target_input,
                last_out_state,
                last_cell_state):
        if isinstance(encoded_sequence, (list, tuple)):
            assert isinstance(encoded_sequence_length, (list, tuple))
            assert len(encoded_sequence) == 1
            assert len(encoded_sequence_length) == 1
            encoded_sequence = encoded_sequence[0]
            encoded_sequence_length = encoded_sequence_length[0]

        if target_input.dim() == 2:
            target_input = target_input.unsqueeze(1)
        elif target_input.dim() == 3:
            pass
        else:
            raise ValueError("argument 'target_input' has shape %s" % str(target_input.shape))

        if last_out_state is None:
            pass
        elif last_out_state.dim() == 2:
            last_out_state = last_out_state.unsqueeze(1)
        elif last_out_state.dim() == 3:
            pass
        else:
            raise ValueError("argument 'last_out_state' has shape %s" % str(last_out_state.shape))

        return encoded_sequence, encoded_sequence_length, target_input, last_out_state, last_cell_state


class LuongAttentionDecoder(RNNSeq2SeqAttentionDecoder):
    def __init__(self, attention, decoder, input_feeding=False, batch_first=True):
        super(LuongAttentionDecoder, self).__init__(attention, decoder, batch_first=batch_first)
        self.i_f = input_feeding

    @property
    def input_feeding(self):
        return self.i_f

    @property
    def input_feeding_dim(self):
        if self.attention.project_out:
            return self.attention.output_dim
        else:
            return self.attention.output_dim + self.attention.query_dim

    @property
    def output_dim(self):
        return self.input_feeding_dim

    def forward(self,
                encoded_sequence,
                encoded_sequence_length,
                target_input,
                last_out_state,
                last_cell_state):
        encoded_sequence, \
        encoded_sequence_length, \
        target_input, \
        last_out_state, \
        last_cell_state = super(LuongAttentionDecoder, self).forward(
            encoded_sequence,
            encoded_sequence_length,
            target_input,
            last_out_state,
            last_cell_state)

        # state_to_attend is not used here, cause it is generated this time step t.
        # ht -> at(attention scores) -> ct(context vector) -> ht_out
        # ht
        if self.input_feeding:
            if last_out_state is None:
                bs, _, _ = target_input.shape
                device = target_input.device
                last_out_state = torch.zeros([bs, 1, self.input_feeding_dim]).to(device)
            target_input = torch.cat([target_input, last_out_state], dim=-1)
        ht, new_cell_state = self.decoder(target_input, last_cell_state)
        # at, ct
        ct, at = self.attention(encoded_sequence, encoded_sequence_length, ht)
        # ht_out
        ht_out = ct if self.attention.project_out else torch.cat([ct, ht], dim=-1)
        return ht_out, ht, new_cell_state, at


class BahdanauAttentionDecoder(RNNSeq2SeqAttentionDecoder):

    @property
    def output_dim(self):
        return self.decoder.output_dim

    def forward(self,
                encoded_sequence,
                encoded_sequence_length,
                target_input,
                last_out_state,
                last_cell_state):
        encoded_sequence, \
        encoded_sequence_length, \
        target_input, \
        last_out_state, \
        last_cell_state = super(BahdanauAttentionDecoder, self).forward(
            encoded_sequence,
            encoded_sequence_length,
            target_input,
            last_out_state,
            last_cell_state)
        # ht-1 -> at -> ct -> ht_out
        # ht-1 -> at,ct
        ct, at = self.attention(encoded_sequence, encoded_sequence_length, last_out_state)
        cell_input = torch.cat([target_input, ct], dim=-1)
        # ht_out
        ht_out, new_cell_state = self.decoder(cell_input, last_cell_state)

        return ht_out, ht_out, new_cell_state, at


class RNNSeq2SeqGenerator(nn.Module):
    def __init__(self, output_layer):
        super(RNNSeq2SeqGenerator, self).__init__()
        self.output_layer = output_layer
        self.sm = nn.LogSoftmax(dim=-1)
        # self.act = nn.Softmax(dim=-1)

    def forward(self, state):
        return self.sm(self.project(state))
        # return self.act(self.output_layer(state))

    def project(self, state):
        return self.output_layer(state)


class RNNSeq2SeqModel(nn.Module):
    def __init__(self,
                 src_embedding,
                 encoder,
                 attention_decoder,
                 tgt_embedding,
                 generator,
                 encoder_initial_layer=None,
                 decoder_initial_layer=None):
        '''

        :param encoder:
        :param decoder:
        :param if_initialize_encoder:
        :param if_initialize_decoder:
        '''
        super(RNNSeq2SeqModel, self).__init__()
        self.src_embedding = src_embedding
        self.encoder = encoder
        self.attention_decoder = attention_decoder
        self.tgt_embedding = tgt_embedding
        self.generator = generator

        self.encoder_initial_layer = encoder_initial_layer
        self.decoder_initial_layer = decoder_initial_layer

    def _reset_parameters(self):
        r"""Initiate parameters.
            Word embeddings and other non-recurrent matrices are initialised
            by sampling from a Gaussian N(0, 0.012),
            recurrent matrices are random orthogonal and
            bias vectors are all initialised to zero.
            Based around the approach from
            "Incorporating Global Visual Features into Attention-Based Neural Machine Translation"
            :cite:`Calixto2017`
        """
        for n, p in self.named_parameters():
            if p.dim() > 1:
                if "rnn" in n:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.uniform_(p.data, -0.01, 0.01)
            else:
                nn.init.constant_(p.data, 0.0)

    def encode(self, input_sequence, sequence_length, encoder_initial_state,
               batch_first=True, enforce_sorted=False):
        input_sequence_embeddings = self.src_embedding(input_sequence)
        packed_input_sequence = pack_padded_sequence(input_sequence_embeddings,
                                                     sequence_length,
                                                     batch_first=batch_first,
                                                     enforce_sorted=enforce_sorted)
        encoded_sequence, encoder_last_state = self.encoder(packed_input_sequence,
                                                            encoder_initial_state)
        padded_encoder_outputs, seq_len = pad_packed_sequence(encoded_sequence,
                                                              batch_first=batch_first)
        # assert seq_len == sequence_length

        return padded_encoder_outputs, sequence_length, encoder_last_state

    def initialize_decoder(self, state):
        if self.decoder_initial_layer is not None:
            return self.attention_decoder.decoder.state_to_initialize(state,
                                                                      self.decoder_initial_layer)
        else:
            zeros = self.attention_decoder.decoder.get_zero_states()
            return zeros

    def forward(self,
                input_sequence,
                input_sequence_length,
                target_sequence,
                target_sequence_length,
                encoder_initial_state=None,
                decoder_initial_state=None,
                batch_first=True,
                enforce_sorted=False,
                generate=True):
        encoded_sequence, encoded_sequence_length, encoder_last_state = self.encode(
            input_sequence,
            input_sequence_length,
            encoder_initial_state,
            batch_first=batch_first,
            enforce_sorted=enforce_sorted)

        encoder_last_state = encoder_last_state.flatten_directions() \
            if encoder_last_state.bidirectional else encoder_last_state
        state_to_initialize_decoder = self.initialize_decoder(
            decoder_initial_state if decoder_initial_state else encoder_last_state)

        outs = train_decoder(self.attention_decoder,
                             self.generator if generate else None,
                             (encoded_sequence,),
                             (encoded_sequence_length,),
                             target_sequence,
                             target_sequence_length,
                             state_to_initialize_decoder,
                             self.tgt_embedding)
        return outs
