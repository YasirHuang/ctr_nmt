# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:utils.py
@time:2020-01-07 
@desc:
'''
import logging
import copy

import torch
import torch.nn as nn


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def anen(*args):
    """
    Assert all arguments are not None
    """
    arguments = (arg for arg in args)
    assert all(arg is not None for arg in arguments), \
        "None object exists: " + str(args)


def todevice(t, device):
    if device:
        return t.to(device)
    else:
        return t


def masked_average(tensor, tensor_lengths, device):
    '''

    :param tensor: The tensor should be at least 2 dimentions: (batch_size, sequence_length,...)
    :param tensor_lengths: batch_size width 1 dimention vector: (batch_size,)
    :param device:
    :return:
    '''
    max_length = torch.max(tensor_lengths)
    mask = generate_mask(tensor_lengths, max_length, device)
    masked_tensor = tensor.masked_fill(mask=mask, value=0.0).to(device)
    masked_tensor = torch.sum(masked_tensor, dim=1)
    masked_tensor = masked_tensor / tensor_lengths.unsqueeze(1)

    return masked_tensor


def generate_mask(sequence_length, max_sequence_length=None, device=None, float_mask=False):
    '''

    :param sequence_length:
    :param max_sequence_length:
    :param device:
    :return:
    '''
    if max_sequence_length:
        max_seq_len = max_sequence_length
    else:
        max_seq_len = max(sequence_length)

    if device is None and hasattr(sequence_length, 'device'):
        device = sequence_length.device
    batch_size = sequence_length.size(0)
    # mask = todevice(torch.arange(max_seq_len).repeat(batch_size, 1), device)
    # print(max_seq_len, batch_size)
    mask = torch.arange(max_seq_len).repeat(batch_size, 1)
    mask = mask.to(device)
    # print(mask)
    # print(max_seq_len, batch_size)
    seq_len_expd = todevice(sequence_length.view(batch_size, 1).repeat(1, max_seq_len), device)
    # print(seq_len_expd)
    # ret_mask = torch.BoolTensor(mask >= seq_len_expd)

    ret_mask = torch.ge(mask, seq_len_expd)
    if float_mask:
        ret_mask = ret_mask.float().masked_fill(mask == 1, float('-inf'))
    # to mask a tensor and do it a softmax
    # att = torch.ones([batch_size, max_seq_len], dtype=torch.float32)
    # masked_att = att.masked_fill(mask=ret_mask, value=-np.inf)
    # nn.Softmax()(masked_att)
    return ret_mask.view(batch_size, max_seq_len, 1)


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden

        triplet_dim = False
        if input.dim() == 3:
            assert input.shape[1] == 1
            triplet_dim = True
            input = input.squeeze(1)

        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        if triplet_dim:
            input = input.unsqueeze(1)
        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        triplet_dim = False
        if input.dim() == 3:
            assert input.shape[1] == 1
            triplet_dim = True
            input = input.squeeze(1)
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        if triplet_dim:
            input = input.unsqueeze(1)
        return input, h_1


def create_rnn_cell(cell_type, cell_layer, input_dim, output_dim, bias=True):
    if cell_type.lower() == 'lstm':
        rnn_cell = nn.LSTMCell
    elif cell_type.lower() == 'gru':
        rnn_cell = nn.GRUCell
    else:
        raise ValueError('Unsupported cell type: %s' % cell_type)
    if cell_layer == 1:

        cell = rnn_cell(input_dim, output_dim, bias)
    else:
        cells_list = [rnn_cell(input_dim, output_dim, bias=bias)]
        for i in range(cell_layer - 1):
            cells_list.append(rnn_cell(output_dim, output_dim, bias=bias))
        cell = nn.ModuleList(cells_list)
    return cell


def create_rnn(cell_type,
               input_dim,
               output_dim,
               num_layers=1,
               dropout=0.0,
               bias=False,
               batch_first=False,
               bidirectional=False,
               custom=False):
    if cell_type.lower() == 'lstm':
        if custom:
            cell = StackedLSTM(num_layers, input_dim, output_dim, dropout)
        else:
            cell = nn.LSTM(input_dim,
                           output_dim,
                           num_layers=num_layers,
                           bias=bias,
                           batch_first=batch_first,
                           dropout=dropout,
                           bidirectional=bidirectional)
    elif cell_type.lower() == 'gru':
        if custom:
            cell = StackedGRU(num_layers, input_dim, output_dim, dropout)
        else:
            cell = nn.GRU(input_dim,
                          output_dim,
                          num_layers=num_layers,
                          bias=bias,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)
    else:
        raise ValueError('Unsupported unit type:%s' % cell_type.lower())
    return cell

def get_activation_layer(activation):
    if not isinstance(activation, str):
        return activation
    if activation.lower() == 'tanh':
        return nn.Tanh()
    elif activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'softmax':
        return nn.Softmax()
    elif activation.lower() == 'log_softmax' or activation.lower() == 'logsoftmax':
        return nn.LogSoftmax()
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError("Unknown activation type %s." % activation)


def train_decoder(attention_decoder,
                  generator,
                  multimodal_inputs,
                  multimodal_input_masks,
                  target_sequence,
                  target_sequence_length,
                  state_to_initial,
                  embedding_layer,
                  **kwargs
                  ):
    assert isinstance(multimodal_inputs, (list, tuple))
    assert isinstance(multimodal_input_masks, (list, tuple))
    input_feeding = False
    if hasattr(attention_decoder, "input_feeding"):
        input_feeding = getattr(attention_decoder, "input_feeding")

    batch_size = multimodal_inputs[0].size(0)
    max_tgt_seq_len = max(target_sequence_length)
    device = multimodal_inputs[0].device
    vocab_size = embedding_layer.vocab_size

    tgt_embs = embedding_layer(target_sequence)

    last_cell_state = state_to_initial
    state_to_attend = last_cell_state.htn

    # predictions = todevice(torch.zeros(batch_size, max_tgt_seq_len, vocab_size), device)
    ht_out = None
    outputs = []
    for t in range(max_tgt_seq_len):
        tgt_input = tgt_embs[:, t, :]

        ht_out, \
        ht_decoder, \
        new_cell_state, \
        h_att_scores = attention_decoder(multimodal_inputs,
                                         multimodal_input_masks,
                                         tgt_input,
                                         ht_out if input_feeding else state_to_attend,
                                         last_cell_state,
                                         **kwargs)
        # preds = generator(ht_out)
        # predictions[:, t, :] = preds.squeeze(1)
        last_cell_state = new_cell_state
        outputs.append(ht_out.squeeze(1))
        state_to_attend = ht_decoder

    outputs = torch.stack(outputs).transpose(0, 1)
    if generator is not None:
        predictions = generator(outputs)
        return predictions
    else:
        return outputs

