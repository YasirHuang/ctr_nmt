# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:utils.py
@time:2020-08-28 
@desc:
'''
import math

import torch
import torch.nn.functional as F
from utils.utils import safe_division

def todevice(t, device):
    if device:
        return t.to(device)
    else:
        return t


def ppl(loss, n_words):
    return math.exp(min(safe_division(loss, n_words), 100))

def expand_to_beam_width(multimodal_inputs, multimodal_input_lengths, beam_width):
    '''

    :param multimodal_inputs:
    :param multimodal_input_lengths:
    :param beam_width:
    :return:
    '''
    batch_size = multimodal_inputs[0].size(0)
    ret_inputs = []
    ret_lengths = []
    # print(multimodal_inputs.shape)
    for i, l in zip(multimodal_inputs, multimodal_input_lengths):
        i_shape = i.shape
        # print(i_shape)
        i = i.view(batch_size,
                   1,
                   i_shape[1],
                   i_shape[2])
        i = i.expand(batch_size,
                     beam_width,
                     i_shape[1],
                     i_shape[2])
        i = i.reshape(batch_size * beam_width,
                      i_shape[1],
                      i_shape[2])
        l = l.unsqueeze(1).repeat(1, beam_width).view(-1)
        ret_inputs.append(i)
        ret_lengths.append(l)

    return tuple(ret_inputs), tuple(ret_lengths)


def beam_search_decode(decoder,
                       generator,
                       multimodal_inputs,
                       multimodal_input_lengths,
                       state_to_initial,
                       embedding_layer,
                       word_to_id_dict,
                       sos='<sos>',
                       eos='<eos>',
                       beam_width=3,
                       max_decode_step=50,
                       **kwargs
                       ):
    '''

    :param multimodal_inputs: (num_modal, (batch_size, max_seq_len, modal_feat_size))
    :param multimodal_input_lengths: (num_modal, (batch_size,))
    :param word_to_id_dict:
    :param sos:
    :param eos:
    :param beam_width:
    :param max_decode_step:
    :return:
    '''

    input_feeding = False
    if hasattr(decoder, "input_feeding"):
        input_feeding = getattr(decoder, "input_feeding")

    if "tgt_vocab_size" in kwargs:
        vocab_size = kwargs["tgt_vocab_size"]
        kwargs.pop("tgt_vocab_size")
    else:
        vocab_size = len(word_to_id_dict)
    batch_size = multimodal_inputs[0].size(0)
    device = multimodal_inputs[0].device

    # expand the input(image_output and audio_output or ...(text)) to beam_width*batch_size
    # after this operation the multimodal_inputs will be like:
    #     (num_modal, (batch_size*beam_width, max_img_seq_len, img_num_unit))
    multimodal_inputs, \
    multimodal_input_lengths = expand_to_beam_width(multimodal_inputs,
                                                    multimodal_input_lengths,
                                                    beam_width)
    state_to_initial = state_to_initial.to_beam(beam_width)

    # initialize h0,c0
    # h0=c0: (k, decoder_num_units)
    cell_state = state_to_initial
    last_cell_state = cell_state
    state_to_attend = cell_state.htn
    ht_out = None

    # preworks for decoding
    k = beam_width * batch_size
    #     k_pre_words_id: (k,sequence_length=1)
    k_pre_words_id = todevice(torch.LongTensor([[word_to_id_dict[sos]]] * k), device)
    #     decoded sequence
    seqs = k_pre_words_id
    #     decoded attention sequence scores
    attention_scores_register = decoder.attention_score_register()

    #     sentence top k scores
    dec_top_k_scores = todevice(torch.zeros(k, 1), device)
    sorted_id_pre = todevice(torch.arange(0, k, beam_width).unsqueeze(1).repeat(1, beam_width).view(-1),
             device)

    complete = torch.BoolTensor([False] * k)
    step = 0
    while True:
        k_pre_words_emb = embedding_layer(k_pre_words_id.squeeze(1))  # (k,1)
        ht_out, \
        ht_decoder, \
        new_cell_state, \
        att_scores = decoder(multimodal_inputs,
                             multimodal_input_lengths,
                             k_pre_words_emb,
                             ht_out if input_feeding else state_to_attend,
                             last_cell_state,
                             **kwargs)

        pred_scores = generator(ht_out)
        pred_scores.squeeze_(1)
        state_to_attend = ht_decoder
        last_cell_state = new_cell_state

        # Is this operation should be "add" ?
        # the answer is yes, because pred_scores are log_softmaxed,
        # a log "add" means a probability "multiply"
        pred_scores = dec_top_k_scores.expand_as(pred_scores) + pred_scores
        pred_scores = torch.reshape(pred_scores, (batch_size, beam_width, vocab_size))

        # (batch_size, beam_width)
        if step == 0:
            dec_top_k_scores, top_k_words_id = pred_scores[:, 0, :].topk(beam_width, -1, True, True)
        else:
            dec_top_k_scores, top_k_words_id = pred_scores.view(batch_size, -1).topk(beam_width, -1, True, True)
        # change to (k)
        dec_top_k_scores = dec_top_k_scores.view(-1)
        top_k_words_id = top_k_words_id.view(-1)

        # Convert unrolled indices to actual indices of scores
        beam_id = top_k_words_id / vocab_size
        sorted_id = sorted_id_pre + beam_id  # (k,1)
        next_word_inds = top_k_words_id % vocab_size  # (k)

        # add new words to sentence
        seqs = torch.cat([seqs[sorted_id], next_word_inds.unsqueeze(1)], dim=1)
        attention_scores_register = decoder.register_attention_score(att_scores, attention_scores_register, sorted_id)

        for i, wid in enumerate(next_word_inds.view(-1)):
            if wid == todevice(torch.tensor(word_to_id_dict[eos]), device):
                complete[i] = True

        if torch.equal(complete, torch.BoolTensor([True]).expand_as(complete)):
            break

        if step > max_decode_step:
            break

        k_pre_words_id = next_word_inds.unsqueeze(1)
        seqs = seqs
        dec_top_k_scores = dec_top_k_scores.unsqueeze(1)
        last_cell_state.resort(sorted_id)
        ht_out = ht_out[sorted_id]
        state_to_attend = state_to_attend[sorted_id]
        # state_to_attend, last_cell_state = decoder.resort_cell_state(last_cell_state, sorted_id)

        step += 1

    return seqs, attention_scores_register


def beam_search_decode_transformer(model,
                                   memory,
                                   src_padding_mask,
                                   word_to_id_dict,
                                   sos='<sos>',
                                   eos='<eos>',
                                   beam_width=3,
                                   max_decode_step=50
                                   ):
    '''

    :param decoder:
    :param logits_projecter:
    :param embedding_layer:
    :param memory:
    :param src_padding_mask:
    :param word_to_id_dict:
    :param sos:
    :param eos:
    :param beam_width:
    :param max_decode_step:
    :return:
    '''

    vocab_size = len(word_to_id_dict)
    device = memory.device

    # expand the input(image_output and audio_output or ...(text)) to beam_width*batch_size
    # after this operation the multimodal_inputs will be like:
    #     (num_modal, (batch_size*beam_width, max_img_seq_len, img_num_unit))

    mem_seq_len = memory.shape[0]
    batch_size = memory.shape[1]
    mem_dim = memory.shape[2]

    memory = memory.view(mem_seq_len,
                         batch_size,
                         1,
                         mem_dim)
    memory = memory.expand(mem_seq_len,
                           batch_size,
                           beam_width,
                           mem_dim)
    memory = memory.reshape(mem_seq_len,
                            batch_size * beam_width,
                            mem_dim)

    assert len(src_padding_mask.shape) == 2
    assert src_padding_mask.shape[0] == batch_size
    assert src_padding_mask.shape[1] == mem_seq_len

    src_padding_mask = src_padding_mask.view(batch_size,
                                             1,
                                             mem_seq_len)
    src_padding_mask = src_padding_mask.expand(batch_size,
                                               beam_width,
                                               mem_seq_len)
    src_padding_mask = src_padding_mask.reshape(batch_size * beam_width,
                                                mem_seq_len)

    # preworks for decoding
    k = beam_width * batch_size
    #     k_pre_words_id: (k, seq_len=1)
    k_pre_words_id = todevice(torch.LongTensor([word_to_id_dict[sos]] * k).unsqueeze(1), device)
    #     decoded sequence
    seqs = k_pre_words_id

    #     sentence top k scores
    dec_top_k_scores = todevice(torch.zeros(k, 1), device)

    complete = torch.BoolTensor([False] * k)
    step = 0
    while True:
        # print(k_pre_words_id)
        preds = model.decode(memory,
                             k_pre_words_id.transpose(0, 1),
                             tgt_key_padding_mask=None,
                             memory_key_padding_mask=src_padding_mask)
        # k_pre_words_emb = embedding_layer(k_pre_words_id)  # (k, seq_len, emb_dim)
        #
        # tgt_seq_len = k_pre_words_emb.size(1)
        # tgt_forward_mask = model._generate_square_subsequent_mask(tgt_seq_len).to(device)
        # # output (tgt_seq_len, batch_size*beam_width, emb_dim)
        # decoder_output = model.transformer_decoder(k_pre_words_emb.transpose(0, 1),
        #                                            memory,
        #                                            tgt_mask=tgt_forward_mask,
        #                                            memory_key_padding_mask=src_padding_mask)
        # preds (tgt_seq_len, batch_size*beam_width, vocab_size)
        # preds = F.log_softmax(logits_projecter(decoder_output), dim=-1)
        # if len(preds) == 3:
        #     print(k_pre_words_id)
        #     print(preds)
        # cur_preds_scores (k, vocab_size)
        cur_preds_scores = preds[-1]
        # Is this operation should be "add" ?
        # "add" previous steps scores to all vocab
        cur_preds_scores = dec_top_k_scores.expand_as(cur_preds_scores) + cur_preds_scores
        cur_preds_scores = torch.reshape(cur_preds_scores, (batch_size, beam_width, vocab_size))

        # (batch_size, beam_width)
        if step == 0:
            dec_top_k_scores, top_k_words_id = cur_preds_scores[:, 0, :].topk(beam_width, -1, True, True)
            # print(k_pre_words_id)
        else:
            dec_top_k_scores, top_k_words_id = cur_preds_scores.view(batch_size, -1).topk(beam_width, -1, True, True)
        # change to (k = batch_size*beam_width)
        dec_top_k_scores = dec_top_k_scores.view(-1)
        top_k_words_id = top_k_words_id.view(-1)

        # Convert unrolled indices to actual indices of scores
        beam_id = top_k_words_id / vocab_size
        # (batch_size)->(batch_size, 1)->(batch_size, beam_width)->(k= batch_size*beam_width)
        sorted_id = todevice(torch.arange(0, k, beam_width).unsqueeze(1).repeat(1, beam_width).view(-1),
                             device) + beam_id  # (k)
        next_word_inds = top_k_words_id % vocab_size  # (k)

        # add new words to sentence
        seqs = torch.cat([seqs[sorted_id], next_word_inds.unsqueeze(1)], dim=1)

        # sort completed beam
        for i, wid in enumerate(next_word_inds.view(-1)):
            if wid == todevice(torch.tensor(word_to_id_dict[eos]), device):
                complete[i] = True

        # if torch.equal(complete, torch.BoolTensor([True]).expand_as(complete)):
        if complete.all():
            break

        if step > max_decode_step:
            break

        k_pre_words_id = seqs
        seqs = seqs
        dec_top_k_scores = dec_top_k_scores.unsqueeze(1)

        step += 1

    return seqs
