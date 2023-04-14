# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:rnn_token_imagine_mmt.py
@time:2020/11/18 
@desc:
'''
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from models.utils import aeq
from models.utils import train_decoder


class MultimodalEmbedding(nn.Module):
    def __init__(self, embedding_layer, project_layer):
        super(MultimodalEmbedding, self).__init__()
        self.embedding_layer = embedding_layer
        self.project_layer = project_layer

    def compound_modalities(self, text_inputs, image_inputs, image_modality_mask):
        '''

        :param text_inputs: shape:if batch_first (batch_size, sequence_length, embedding_dim),
        :param image_inputs: shape:if batch_first (batch_size, sequence_length, embedding_dim),
        :param image_modality_mask: shape:if batch_first (batch_size, sequence_length), filled with 0/1
        :return: ~mask * t + mask * i
        '''
        aeq(text_inputs.shape[0], image_inputs.shape[0], image_modality_mask.shape[0])
        aeq(text_inputs.shape[1], image_inputs.shape[1], image_modality_mask.shape[1])
        aeq(text_inputs.shape[2], image_inputs.shape[2])

        mask = image_modality_mask.float().unsqueeze(-1)
        antimask = 1.0 - mask
        return antimask * text_inputs + mask * image_inputs

    def forward(self, text_input_sequence, image_input_sequence=None, image_sequence_mask=None):
        embs = self.embedding_layer(text_input_sequence)
        if image_input_sequence is not None:
            assert image_sequence_mask is not None
            embedded_img = self.project_layer(image_input_sequence)
            embs = self.compound_modalities(embs, embedded_img, image_sequence_mask)

        return embs


class RNNTokenImagineModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,
                 src_embedding,
                 encoder,
                 attention_decoder,
                 tgt_embedding,
                 generator,
                 encoder_initial_layer=None,
                 decoder_initial_layer=None):
        super(RNNTokenImagineModel, self).__init__()
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

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
    #     return mask

    def encode(self, input_sequence, sequence_length, image_sequence, image_mask,
               encoder_initial_state,
               batch_first=True, enforce_sorted=False):
        src_embs = self.src_embedding(input_sequence, image_sequence, image_mask)
        packed_input_sequence = pack_padded_sequence(src_embs,
                                                     sequence_length,
                                                     batch_first=batch_first,
                                                     enforce_sorted=enforce_sorted)
        encoded_sequence, encoder_last_state = self.encoder(packed_input_sequence,
                                                            encoder_initial_state)
        padded_encoder_outputs, seq_len = pad_packed_sequence(encoded_sequence,
                                                              batch_first=batch_first)
        return padded_encoder_outputs, sequence_length, encoder_last_state

    def initialize_decoder(self, state):
        if self.decoder_initial_layer is not None:
            return self.attention_decoder.decoder.state_to_initialize(state,
                                                                      self.decoder_initial_layer)
        else:
            zeros = self.attention_decoder.decoder.get_zero_states()
            return zeros

    def decode(self, memory, tgt, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_seq_len = tgt.size(0)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

        tgt_embs = self.tgt_embedding_layer(tgt)
        decoder_output = self.transformer_decoder(tgt_embs,
                                                  memory,
                                                  tgt_mask=tgt_mask,
                                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask)

        logits = self.generator_layer(decoder_output)
        return logits

    def forward(self,
                input_sequence,
                input_sequence_length,
                image_sequence,
                image_mask,
                target_sequence,
                target_sequence_length,
                encoder_initial_state=None,
                decoder_initial_state=None,
                batch_first=True,
                enforce_sorted=False,
                generate=True):
        encoded_sequence, encoded_sequence_length, encoder_last_state = self.encode(
            input_sequence, input_sequence_length, image_sequence, image_mask,
            encoder_initial_state, batch_first=batch_first, enforce_sorted=enforce_sorted)

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


class DualModule(nn.Module):
    def __init__(self, singlemodal_model, multimodal_model):
        super(DualModule, self).__init__()
        self.singlemodal_model = singlemodal_model
        self.multimodal_model = multimodal_model

    @property
    def singlemodal(self):
        return self.singlemodal_model
    @property
    def multimodal(self):
        return self.multimodal_model

    #
    # def singlemodal(self,
    #                 input_sequence,
    #                 input_sequence_length,
    #                 target_sequence,
    #                 target_sequence_length,
    #                 encoder_initial_state=None,
    #                 decoder_initial_state=None,
    #                 batch_first=True,
    #                 enforce_sorted=False,
    #                 generate=True):
    #     return self.singlemodal_model(
    #         input_sequence,
    #         input_sequence_length,
    #         target_sequence,
    #         target_sequence_length,
    #         encoder_initial_state=encoder_initial_state,
    #         decoder_initial_state=decoder_initial_state,
    #         batch_first=batch_first,
    #         enforce_sorted=enforce_sorted,
    #         generate=generate)
    #
    # def multimodal(self,
    #                input_sequence,
    #                input_sequence_length,
    #                image_sequence,
    #                image_mask,
    #                target_sequence,
    #                target_sequence_length,
    #                encoder_initial_state=None,
    #                decoder_initial_state=None,
    #                batch_first=True,
    #                enforce_sorted=False,
    #                generate=True):
    #     return self.multimodal_model(
    #         input_sequence,
    #         input_sequence_length,
    #         image_sequence,
    #         image_mask,
    #         target_sequence,
    #         target_sequence_length,
    #         encoder_initial_state=encoder_initial_state,
    #         decoder_initial_state=decoder_initial_state,
    #         batch_first=batch_first,
    #         enforce_sorted=enforce_sorted,
    #         generate=generate)
