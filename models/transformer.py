# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:transformer.py
@time:2020-09-04 
@desc:
'''
import math

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, output_layer):
        super(Generator, self).__init__()
        self.output_layer = output_layer
        self.sm = nn.LogSoftmax(dim=-1)
        # self.act = nn.Softmax(dim=-1)

    def forward(self, state):
        return self.sm(self.project(state))
        # return self.act(self.output_layer(state))

    def project(self, state):
        return self.output_layer(state)

class PositionalEncoder(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoder(d_model)
    """

    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 embedding_layer,
                 positional_encoder):
        super(PositionalEmbedding, self).__init__()
        self.embedding_layer = embedding_layer
        self.positional_encoder = positional_encoder

    def forward(self, input_sequence):
        embs = self.embedding_layer(input_sequence)
        emb_dim = embs.shape[-1]
        embs = embs * math.sqrt(emb_dim)
        posed_embs = self.positional_encoder(embs)
        return posed_embs


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,
                 src_embedding_layer,
                 transformer_encoder,
                 transformer_decoder,
                 tgt_embedding_layer,
                 generator_layer):
        super(TransformerModel, self).__init__()
        self.src_embedding_layer = src_embedding_layer
        self.transformer_encoder = transformer_encoder
        self.transformer_decoder = transformer_decoder
        self.tgt_embedding_layer = tgt_embedding_layer
        self.generator_layer = generator_layer

        # self._reset_parameters()

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

    def encode(self, src, mask):
        src_embs = self.src_embedding_layer(src)
        encoded_src = self.transformer_encoder(src_embs, src_key_padding_mask=mask)
        return encoded_src

    def decode(self, memory, tgt, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_seq_len = tgt.size(0)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        # tgt_forward_mask = Transformer.generate_square_subsequent_mask(Transformer,tgt_seq_len).to(tgt.device)
        # tgt_forward_mask = tgt_forward_mask.transpose(0,1)

        tgt_embs = self.tgt_embedding_layer(tgt)
        decoder_output = self.transformer_decoder(tgt_embs,
                                                  memory,
                                                  tgt_mask=tgt_mask,
                                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask)

        logits = self.generator_layer(decoder_output)
        return logits

    def forward(self, src, src_padding_mask, tgt, tgt_padding_mask):
        memory = self.encode(src, src_padding_mask)

        logits = self.decode(memory,
                             tgt,
                             tgt_key_padding_mask=tgt_padding_mask,
                             memory_key_padding_mask=src_padding_mask)

        return logits
