# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:basic_models.py
@time:2020-08-27 
@desc:
'''
import numpy as np
import copy
import logging

import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

from models.utils import create_rnn, generate_mask, aeq, anen, get_activation_layer


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=-1.0, bias=True):
        super(LinearLayer, self).__init__()

        self.linear_layer = nn.Linear(input_dim, output_dim, bias=bias)
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.dropout = dropout

    @property
    def output_dim(self):
        return self.out_dim

    @property
    def input_dim(self):
        return self.in_dim

    def forward(self, inputs):
        if self.dropout > 0:
            return self.dropout_layer(self.linear_layer(inputs))
        else:
            return self.linear_layer(inputs)


class BahdanauAttention(nn.Module):
    def __init__(self,
                 encoder_dim,
                 decoder_dim,
                 attention_dim,
                 normalization=False):
        super(BahdanauAttention, self).__init__()
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=normalization)
        self.query_layer = nn.Linear(decoder_dim, attention_dim, bias=normalization)
        if normalization:
            self.project_layer = weight_norm(nn.Linear(attention_dim, 1))
        else:
            self.project_layer = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, encoded_sequence_length, decoder_out):
        # encoder_out: (batch_size, sequence_length, encoder_dim)
        # decoder_out: (batch_size, decoder_dim)

        # W1*hti (+b1)
        keys = self.memory_layer(encoder_out)  # (batch_size, sequence_length, attention_dim)
        # W2*st (+b2)
        processed_query = self.query_layer(decoder_out)  # (batch_size, attention_dim)
        unsqueezed_processed_query = processed_query.unsqueeze(1)  # (batch_size, 1, attention_dim)
        # eti = V*tanh(W1*hti+W2*st (+b1+b2))
        score = self.project_layer(self.tanh(keys + unsqueezed_processed_query))  # (batch_size, sequence_length, 1)
        max_sequence_length = encoder_out.size(1)
        att_mask = generate_mask(encoded_sequence_length, max_sequence_length, encoder_out.device)
        # αti = softmax（eti)
        att_scores = self.softmax(score.masked_fill(mask=att_mask, value=-np.inf))  # (batch_size, sequence_length, 1)
        context = (att_scores * encoder_out).sum(dim=1)
        return context, att_scores.squeeze(2)


class GlobalAttention(nn.Module):
    def __init__(self,
                 context_dim,
                 query_dim,
                 attention_dim,
                 attention_type='bahdanau',
                 project_out=False,
                 dropout=-1.0):
        '''

        :param context_dim:
        :param query_dim:
        :param attention_dim:
        :param attention_type: bahdanau | normed_bahdanau | dot_luong | general_luong/luong
        '''
        super(GlobalAttention, self).__init__()
        self.ctx_dim = context_dim
        self.q_dim = query_dim
        self.att_dim = attention_dim
        self.attention_type = attention_type.lower()
        self.project_out = project_out
        self.dropout = dropout
        self.activate_projected_out = False

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else lambda x: x

        if self.attention_type.endswith('bahdanau'):
            norm = False if self.attention_type == 'normed_bahdanau' else True
            self.memory_layer = nn.Linear(context_dim, attention_dim, bias=norm)
            self.query_layer = nn.Linear(query_dim, attention_dim, bias=norm)
            self.v = weight_norm(nn.Linear(attention_dim, 1)) if norm else \
                nn.Linear(attention_dim, 1)
        elif self.attention_type == 'general_luong' or self.attention_type == 'luong':
            self.w = nn.Linear(query_dim, context_dim, bias=False)

        if project_out:
            # bahdanau wants it with bias
            out_bias = self.attention_type.endswith('bahdanau')
            print("attention_dim", attention_dim)
            self.out_projector = LinearLayer(query_dim + context_dim,
                                             attention_dim,
                                             bias=out_bias)
            if self.attention_type.endswith('luong'):
                self.activate_projected_out = True

    @property
    def context_dim(self):
        return self.ctx_dim

    @property
    def query_dim(self):
        return self.q_dim

    @property
    def attention_dim(self):
        return self.att_dim

    @property
    def output_dim(self):
        if self.project_out:
            return self.out_projector.output_dim
        else:
            return self.ctx_dim

    def _score(self, context, q):
        '''

        :param context: (bs, src_seq_len, ctx_dim)
        :param q: (bs, tgt_seq_len=1, query_dim)
        :return:
        '''
        if self.attention_type.endswith('bahdanau'):
            keys = self.memory_layer(context)  # (bs, src_seq_len, att_dim)
            processed_q = self.query_layer(q)  # (bs, tgt_seq_len=1, att_dim)
            # (bs, seq_len, 1, att_dim) + (bs, 1, tgt_seq_len=1, att_dim) -> (bs, seq_len, tgt_seq_len=1, 1)
            scores = self.v(self.tanh(keys.unsqueeze(2) + processed_q.unsqueeze(1))).squeeze(-1)
        elif self.attention_type == 'general_luong' or self.attention_type == 'luong':
            W_q = self.w(q)  # (bs, tgt_seq_len=1, ctx_dim)
            W_q = W_q.transpose(-1, -2)  # (bs, ctx_dim, tgt_seq_len=1)
            # (bs,seq_len, ctx_dim) * (bs, ctx_dim, tgt_seq_len=1) -> (bs, seq_len, tgt_seq_len=1)
            scores = torch.bmm(context, W_q)
        elif self.attention_type == 'dot_luong':
            # (bs, seq_len, ctx_dim) * (bs, query_dim, tgt_seq_len=1) -> (bs, seq_len, tgt_seq_len=1)
            assert context.shape[-1] == q.shape[-1]
            scores = torch.bmm(context, q.transpose(-1, -2))
        else:
            raise ValueError("Unsupported attention type: %s." % self.attention_type)
        return scores

    def forward(self, context, context_mask, query):
        '''

        :param context: (bs, seq_len, ctx_dim)
        :param context_mask: (bs, seq_len, tgt_seq_len=1)
        :param query: (bs, tgt_seq_len=1, query_dim) or None
        :return: context vector c(bs, tgt_seq_len=1, ctx_dim),
                    scores(bs, ctx_seq_len, tgt_seq_len=1)
        '''
        # check params
        cbs, src_len, ctx_dim = context.shape

        if query is None:
            bs = context.shape[0]
            device = context.device
            query = torch.zeros([bs, 1, self.query_dim]).to(device)
        else:
            qbs, tgt_len, q_dim = query.shape
            aeq(cbs, qbs)

        scores = self._score(context, query)  # (bs, ctx_seq_len, tgt_seq_len=1)
        sbs, s_src_len, s_tgt_len = scores.shape
        aeq(sbs, cbs)
        aeq(s_src_len, src_len)

        masked_scores = scores.masked_fill(mask=context_mask, value=-np.inf)
        normed_scores = self.softmax(masked_scores)
        # (bs, tgt_seq_len=1, ctx_seq_len) * (bs, ctx_seq_len, ctx_dim) -> (bs, tgt_seq_len=1, ctx_dim)
        c = torch.bmm(normed_scores.transpose(-1, -2), context)

        if self.project_out:
            cated_c = torch.cat([c, query], dim=-1)
            c = self.out_projector(cated_c)
            c = self.tanh(c) if self.activate_projected_out else c
            c = self.dropout(c)
        return c, normed_scores


class RNNCellState(object):
    def __init__(self,
                 cell_hidden=None,
                 cell_info=None):
        '''

        :param cell_input: (num_layers * num_directions, batch_size, input_size)
        :param cell_hidden: tuple if rnncell is a lstm,
                            else (batch_size(, layer_number if number_of_layer > 1), hidden_size)
        '''

        self.cell_info = cell_info
        self.hidden = None

        self.h = cell_hidden

    def cell_zero_states(self, batch_size):
        num_layers = self.cell_info['num_layers']
        num_directions = self.cell_info['num_directions']
        hidden_size = self.cell_info['hidden_size']
        return torch.zero([num_layers * num_directions, batch_size, hidden_size])

    @property
    def batch_size(self):
        if self.h is None:
            return None
        else:
            nld, bs, hs = self.h.shape
            return bs

    @property
    def h(self):
        if self.hidden is None:
            return None
        aeq(self.hidden.dim(), 3)
        nld, bs, hs = self.hidden.shape
        cnl, cnd, chs = self.cell_infomation()
        aeq(cnl * cnd, nld)
        aeq(hs, chs)
        return self.hidden.contiguous()

    @h.setter
    def h(self, cell_hidden):
        num_layers, num_directions, hidden_size = self.cell_infomation()
        is_none_state = False
        if cell_hidden is None:
            is_none_state = True
        else:
            assert num_layers is not None
            assert num_directions is not None
            assert hidden_size is not None
            if len(cell_hidden.shape) == 4:
                nl, nd, bs, hs = cell_hidden.shape
                aeq(num_layers, nl)
                aeq(num_directions, nd)
                aeq(hidden_size, hs)
                nld = num_layers * num_directions
                cell_hidden = cell_hidden.view(nld, bs, hs)
            elif len(cell_hidden.shape) == 3:
                nld, bs, hs = cell_hidden.shape
                aeq(nld, num_layers * num_directions)
                aeq(hs, hidden_size)
                if self.batch_size is not None:
                    aeq(self.batch_size, bs)
            elif len(cell_hidden.shape) == 2:
                logging.warning("RNN cell hidden is 2 dimensional tensor. "
                                "The 0 dim will be treat as batch size, "
                                "and 1 dim will be treat as hidden size.")
                bs, hs = cell_hidden.shape
                aeq(hs, hidden_size)
                nld = num_layers * num_directions
                cell_hidden = cell_hidden.unsqueeze(0).repeat(nld, 1, 1)
            else:
                raise ValueError("Illegal hidden shape: %s." % str(cell_hidden.shape))

        if is_none_state:
            self.hidden = None
        else:
            self.hidden = cell_hidden

    def __call__(self):
        return self.get_cell_state()

    def cell_infomation(self):
        num_layers = self.cell_info['num_layers']
        num_directions = self.cell_info['num_directions']
        hidden_size = self.cell_info['hidden_size']
        return num_layers, num_directions, hidden_size

    @property
    def bidirectional(self):
        _, directions, _ = self.cell_infomation()
        if directions == 2:
            return True
        else:
            aeq(directions, 1)
            return False

    @property
    def htn(self):
        '''
        represents the last layer state of h
        :return:
        '''
        nl, nd, hs = self.cell_infomation()
        if self.hidden is not None:
            last = self.h.view(nl, nd, -1, hs)[-1]
            last = last.transpose(0, 1).contiguous().view(-1, nd * hs)
            return last
        else:
            return None

    def resort(self, sorted_id):
        if self.h is None:
            return
        assert self.h.dim() == 3
        self.h = self.h[:, sorted_id, :]  # need to be tested

    def get_cell_state(self):
        return self.h

    def new_state(self, state):
        num_layers, num_directions, hidden_size = self.cell_infomation()
        cell_info = {"num_layers": num_layers,
                     "num_directions": num_directions,
                     "hidden_size": hidden_size}
        if state is None:
            return RNNCellState(None, cell_info)

        n_l_d, bs, hs = state.shape
        aeq(num_layers * num_directions, n_l_d)
        aeq(hidden_size, hs)

        return RNNCellState(state, cell_info)

    def copy(self, times):
        return [copy.deepcopy(self) for _ in range(times)]

    def to(self, device):
        if self.h is None:
            pass
        else:
            self.h = self.h.to(device)

    @staticmethod
    def _to_beam(vector, beam_size):
        h = vector
        nld, bs, hs = h.shape
        h = h.view(nld, bs, 1, hs)
        h = h.expand(nld, bs, beam_size, hs)
        return h.reshape(nld, bs * beam_size, hs)

    def to_beam(self, beam_size):
        if self.h is not None:
            beamed_h = self._to_beam(self.h, beam_size)
            return self.new_state(beamed_h)
        else:
            return self.new_state(None)

    def flatten_directions(self):
        nl, nd, hs = self.cell_infomation()
        _, bs, _ = self.h.shape
        h = self.h.view(nl, nd, bs, hs).transpose(1, 2).contiguous().view(nl, bs, nd * hs)
        cell_info = {"num_layers": nl, "num_directions": 1, "hidden_size": nd * hs}
        return type(self)(h, cell_info)

    def project(self, layer):
        assert layer is not None
        nl, nd, hs = self.cell_infomation()
        cell_info = {"num_layers": nl,
                     "num_directions": nd,
                     "hidden_size": hs}
        if self.h is None:
            cell_info["hidden_size"] = layer.output_dim if hasattr(layer, "output_dim") else hs
            return type(self)(None, cell_info)

        h = layer(self.h)
        new_nld, new_bs, new_hs = h.shape
        cell_info["hidden_size"] = new_hs
        return type(self)(h, cell_info)


class GRUCellState(RNNCellState):

    def new_state(self, state):
        num_layers, num_directions, hidden_size = self.cell_infomation()
        cell_info = {"num_layers": num_layers,
                     "num_directions": num_directions,
                     "hidden_size": hidden_size}
        if state is None:
            return GRUCellState(None, cell_info)

        n_l_d, bs, hs = state.shape
        aeq(num_layers * num_directions, n_l_d)
        aeq(hidden_size, hs)

        return GRUCellState(state, cell_info)


class LSTMCellState(RNNCellState):
    def __init__(self, h=None, c=None,
                 cell_info=None):
        super(LSTMCellState, self).__init__(
            h,
            cell_info
        )
        if c is None:
            self.c_hidden = self.h
        else:
            self.c_hidden = c

    @property
    def c(self):
        if self.c_hidden is None:
            return None
        nld, bs, hs = self.c_hidden.shape
        h_nld, h_bs, h_hs = self.h.shape
        aeq(nld, h_nld)
        aeq(bs, h_bs)
        aeq(hs, h_hs)

        return self.c_hidden.contiguous()

    @c.setter
    def c(self, c):
        self.c_hidden = c

    def __call__(self):
        return self.get_cell_state()

    def resort(self, sorted_id):
        if self.h is None:
            return
        assert self.h.dim() == 3
        assert self.c.dim() == 3
        self.hidden = self.h[:, sorted_id, :]  # need to be tested
        self.c = self.c[:, sorted_id, :]  # need to be tested

    def get_cell_state(self):
        if self.h is None:
            return None
        return self.h, self.c

    def new_state(self, state):
        num_layers, num_directions, hidden_size = self.cell_infomation()
        cell_info = {"num_layers": num_layers,
                     "num_directions": num_directions,
                     "hidden_size": hidden_size}
        if state is None:
            return LSTMCellState(cell_info=cell_info)

        assert isinstance(state, (tuple, list))
        h, c = state
        h_nld, h_bs, h_hs = h.shape
        c_nld, c_bs, c_hs = c.shape
        aeq(h_nld, c_nld, num_layers * num_directions)
        aeq(h_bs, c_bs)
        aeq(h_hs, c_hs, hidden_size)

        return LSTMCellState(h, c, cell_info)

    def to(self, device):
        if self.h is None:
            pass
        else:
            self.h = self.h.to(device)
            self.c = self.c.to(device)

    def to_beam(self, beam_size):
        if self.h is not None:
            anen(self.c)
            beamed_h = self._to_beam(self.h, beam_size)
            beamed_c = self._to_beam(self.c, beam_size)
            state = (beamed_h, beamed_c)
            return self.new_state(state)
        else:
            return self.new_state(None)

    def flatten_directions(self):
        nl, nd, hs = self.cell_infomation()
        cell_info = {"num_layers": nl, "num_directions": 1, "hidden_size": nd * hs}
        _, bs, _ = self.h.shape
        h = self.h
        c = self.c
        if h is not None:
            assert self.c is not None
            h = h.view(nl, nd, bs, hs).transpose(1, 2).contiguous().view(nl, bs, nd * hs)
            c = c.view(nl, nd, bs, hs).transpose(1, 2).contiguous().view(nl, bs, nd * hs)
        return type(self)(h, c, cell_info)

    def project(self, layer):
        assert layer is not None
        nl, nd, hs = self.cell_infomation()
        cell_info = {"num_layers": nl,
                     "num_directions": nd,
                     "hidden_size": hs}
        if self.h is None:
            cell_info["hidden_size"] = layer.output_dim if hasattr(layer, "output_dim") else hs
            return type(self)(cell_info=cell_info)

        h = layer(self.h)
        c = layer(self.c)
        new_nld, new_bs, new_hs = h.shape
        cell_info["hidden_size"] = new_hs
        return type(self)(h, c, cell_info=cell_info)


class RNN(nn.Module):
    def __init__(self,
                 layer_type,
                 layer_number,
                 input_dim,
                 output_dim,
                 dropout=0.0,
                 batch_first=False,
                 bidirectional=False,
                 bias=False,
                 custom=False):
        '''

        :param layer_type:
        :param layer_number:
        :param input_dim:
        :param output_dim:
        :param bias:
        '''
        super(RNN, self).__init__()
        self.rnn = create_rnn(layer_type,
                              input_dim,
                              output_dim,
                              num_layers=layer_number,
                              dropout=dropout,
                              bias=bias,
                              batch_first=batch_first,
                              bidirectional=bidirectional,
                              custom=custom
                              )
        self.layer_type = layer_type
        self.layer_number = layer_number
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.bias = bias
        self.custom = custom

    @property
    def directions(self):
        if self.bidirectional:
            return 2
        else:
            return 1

    @property
    def output_dim(self):
        return self.out_dim * self.directions

    @property
    def input_dim(self):
        return self.in_dim

    @property
    def cell_info(self):
        directions = 2 if self.bidirectional else 1
        cell_info = {"num_layers": self.layer_number,
                     "num_directions": directions,
                     "hidden_size": self.out_dim}
        return cell_info

    def get_zero_states(self):
        '''
        zero states is actually None
        :return:
        '''
        if self.layer_type.lower() == 'gru':
            state = GRUCellState(cell_info=self.cell_info)
        elif self.layer_type.lower() == 'lstm':
            state = LSTMCellState(cell_info=self.cell_info)
        else:
            raise ValueError("Unsupported layer type %s." % self.layer_type)
        return state

    def state_to_initialize(self, state, layer=None):
        if isinstance(state, RNNCellState):
            if layer is not None:
                state = state.project(layer)
            return state
        if layer is not None:
            state_to_initialize = layer(state)
        else:
            state_to_initialize = state

        if self.layer_type.lower() == 'lstm':
            state = GRUCellState(state_to_initialize, cell_info=self.cell_info)
        elif self.layer_type.lower() == 'gru':
            state = LSTMCellState(state_to_initialize, cell_info=self.cell_info)
        return state

    def forward(self, input, state_0: RNNCellState = None):
        '''

        :param input_sequence: (batch_size, seq_len, emb_dim) or a packed sequence,
        when seq_len=1, it is an one step rnn
        :param last_time_state:
        :return:
        '''
        # arg check
        if isinstance(input, nn.utils.rnn.PackedSequence):
            pass
        else:
            if input.dim() == 2:
                bs, emb_dim = input.shape
                seq_len = 1
                input = input.unsqueeze(1)
            elif input.dim() == 3:
                bs, seq_len, emb_dim = input.shape
            else:
                raise ValueError("Invalid input shape for RNN(%s): %s" %
                                 (self.layer_type, str(input.shape)))
        if state_0 is None:
            state_0 = self.get_zero_states()
        output, state_n = self.rnn(input,
                                   state_0())
        state_n = state_0.new_state(state_n)
        return output, state_n


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=-1.0, pad_token_id=None):
        super(EmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.pad_token_id = pad_token_id

    def forward(self, inputs):
        assert all((inputs >= 0).view(-1))
        assert all((inputs < self.vocab_size).view(-1))
        if self.dropout > 0:
            return self.dropout_layer(self.embedding(inputs))
        else:
            return self.embedding(inputs)

    def load_embeddings_from_numpy(self, embedding_path, embedding_word_dict_path, source_vocab_dict, pad_token):
        import json

        emb = np.load(embedding_path)
        with open(embedding_word_dict_path, 'r') as fp:
            embedding_word_dict = json.load(fp)
        sorted_vocab_keys = sorted(source_vocab_dict.keys(), key=lambda x: source_vocab_dict[x])

        ordered_emb = []
        for k in sorted_vocab_keys:
            if k in embedding_word_dict:
                ordered_emb.append(emb[embedding_word_dict[k]])
            elif k == pad_token:
                ordered_emb.append(np.zeros_like(emb[0]))
            else:
                raise ValueError("Key word %s not found in embedding_word_dict" % k)
        ordered_emb = np.array(ordered_emb)
        self.embedding.weight.data = torch.from_numpy(ordered_emb).float().to(self.embedding.weight.device)
        print("src_embedding loaded from %s." % embedding_path)


class ProjectLayer(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, activation=None, bias=False, dropout=0.0, is_big=False):
        super(ProjectLayer, self).__init__()

        assert num_layers > 0 or dropout > 0.0

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.out_dim = output_dim if num_layers > 0 else input_dim
        self.activation = activation
        self.bias = bias
        self.dropout = dropout
        self.is_big = is_big

        if num_layers > 0:
            self.model = nn.Sequential()
            in_dim = input_dim
            out_dim = input_dim if is_big else output_dim
            for i in range(num_layers - 1):
                self.model.add_module("image_project_layer_%d" % i, nn.Linear(in_dim, out_dim, bias=bias))
                in_dim = out_dim
                if activation is not None:
                    self.model.add_module("%s_%d" % (activation, i), get_activation_layer(activation))

            self.model.add_module("image_project_layer_%d" % (num_layers - 1),
                                  nn.Linear(in_dim, output_dim, bias=bias))
            if activation is not None:
                self.model.add_module("%s_%d" % (activation, (num_layers - 1)), get_activation_layer(activation))
            if dropout > 0.0:
                self.model.add_module("dropout_layer", nn.Dropout(dropout))

        elif num_layers == 0:
            self.model = nn.Dropout(dropout)

    @property
    def output_dim(self):
        return self.out_dim

    def forward(self, image):
        return self.model(image)


def build_project_layer(num_layer, input_dim, output_dim,
                        activation=None, bias=False, dropout=0.0, is_big=False):
    if num_layer > 0 or dropout > 0.0:
        project_layer = ProjectLayer(
            num_layer,
            input_dim,
            output_dim,
            activation=activation,
            bias=bias,
            dropout=dropout,
            is_big=is_big)
    else:
        project_layer = lambda x: x
    return project_layer
