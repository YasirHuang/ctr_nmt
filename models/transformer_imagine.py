# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:transformer_imagine.py
@time:2020-09-20 
@desc:
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import aeq

class MultimodalEmbedding(nn.Module):
    def __init__(self, embedding_layer, project_layer, positional_encoder):
        super(MultimodalEmbedding, self).__init__()
        self.embedding_layer = embedding_layer
        self.project_layer = project_layer
        self.positional_encoder = positional_encoder

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

    def _embed(self, text_input_sequence, image_input_sequence=None, image_sequence_mask=None):
        embs = self.embedding_layer(text_input_sequence)
        if image_input_sequence is not None:
            assert image_sequence_mask is not None
            embedded_img = self.project_layer(image_input_sequence)
            embs = self.compound_modalities(embs, embedded_img, image_sequence_mask)

        emb_dim = embs.shape[-1]
        return embs * math.sqrt(emb_dim)

    def forward(self, text_input_sequence, image_input_sequence=None, image_sequence_mask=None):
        embs = self._embed(text_input_sequence, image_input_sequence, image_sequence_mask)

        posed_embs = self.positional_encoder(embs)
        return posed_embs

class TransformerImagineModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,
                 src_embedding_layer,
                 transformer_encoder,
                 transformer_decoder,
                 tgt_embedding_layer,
                 generator_layer):
        super(TransformerImagineModel, self).__init__()
        self.src_embedding_layer = src_embedding_layer
        self.transformer_encoder = transformer_encoder
        self.transformer_decoder = transformer_decoder
        self.tgt_embedding_layer = tgt_embedding_layer
        self.generator_layer = generator_layer

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
    #     return mask

    def _generate_square_subsequent_mask(self, sz):
        # down triangle False matrix, like:
        # [[False,  True,  True],
        #  [False, False,  True],
        #  [False, False, False]]
        mask = (torch.triu(torch.ones(sz, sz)) == 0).transpose(0, 1)
        # for zeros matrix, the True positions will be filled with '-inf'
        mask = torch.zeros_like(mask, dtype=torch.float).masked_fill(mask, float('-inf'))
        return mask

    def encode(self, src, mask, img=None, img_mask=None):
        src_embs = self.src_embedding_layer(src, img, img_mask)
        encoded_src = self.transformer_encoder(src_embs, src_key_padding_mask=mask)
        return encoded_src

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

    def forward(self, src, src_padding_mask, tgt, tgt_padding_mask, img=None, img_mask=None):
        memory = self.encode(src, src_padding_mask, img, img_mask)

        logits = self.decode(memory,
                             tgt,
                             tgt_key_padding_mask=tgt_padding_mask,
                             memory_key_padding_mask=src_padding_mask)

        return logits

class DualModule(nn.Module):
    def __init__(self, transformer, multimodal_transformer):
        super(DualModule, self).__init__()
        self.transformer = transformer
        self.multimodal_transformer = multimodal_transformer


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def singlemodal(self, src, src_padding_mask, tgt, tgt_padding_mask):
        return self.transformer(src, src_padding_mask, tgt, tgt_padding_mask)

    def multimodal(self, src, src_padding_mask, tgt, tgt_padding_mask, img=None, img_mask=None):
        return self.multimodal_transformer(src, src_padding_mask, tgt, tgt_padding_mask, img, img_mask)