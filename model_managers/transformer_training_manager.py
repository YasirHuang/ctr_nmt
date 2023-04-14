# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:rnn_seq2seq_training_manager.py
@time:2020-08-28 
@desc:
'''
import os
import sys
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import LayerNorm
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerDecoder
from torch.nn import TransformerDecoderLayer

from models.basic_models import EmbeddingLayer, LinearLayer
from models.transformer import PositionalEncoder
from models.transformer import PositionalEmbedding
from models.transformer import TransformerModel
from models.transformer import Generator
from models.utils import generate_mask

from model_managers.utils import beam_search_decode_transformer
from model_managers.basic_training_manager import BasicTrainingManager

from utils.config import MODE


class TransformerTrainingManager(BasicTrainingManager):
    def __init__(self, mode: MODE, FLAGS, log=True):
        super(TransformerTrainingManager, self).__init__(mode, FLAGS, log)

        # create model parameters
        self.share_embedding = FLAGS.share_embedding
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
        # data parameters
        self.max_len = FLAGS.max_len
        self.src_vocab_size = FLAGS.src_vocab_size  # requires dataset to be prepared
        self.tgt_vocab_size = FLAGS.tgt_vocab_size  # requires dataset to be prepared
        self.batch_size = FLAGS.batch_size
        self.batch_first = FLAGS.batch_first
        # training parameters
        self.pretrained_embedding = FLAGS.pretrained_embedding  # specify in shell script
        self.multiple_gpu = FLAGS.multiple_gpu

        # pretrained embedding
        self.source_embeddings_path = FLAGS.source_embeddings_path
        self.embedding_word_dict_path = FLAGS.embedding_word_dict_path
        # others
        self.pad_token = FLAGS.pad
        self.pad_token_id = FLAGS.pad_token_id

    def create_encoder(self,
                       src_vocab_size,
                       src_max_len,
                       embedding_dim,
                       encoder_num_layers,
                       encoder_dim,
                       number_of_head,
                       embedding_dropout,
                       encoder_dropout,
                       activation='relu'):

        src_embedding_layer = EmbeddingLayer(src_vocab_size, embedding_dim)
        positional_encoder = PositionalEncoder(model_dim=embedding_dim,
                                               dropout=embedding_dropout,
                                               max_len=src_max_len * 2)
        positional_embedding = PositionalEmbedding(src_embedding_layer, positional_encoder)

        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim,
                                                nhead=number_of_head,
                                                dim_feedforward=encoder_dim,
                                                dropout=encoder_dropout,
                                                activation=activation)
        layer_norm = LayerNorm(embedding_dim)
        encoder = TransformerEncoder(encoder_layer, encoder_num_layers, layer_norm)
        return positional_embedding, encoder

    def create_decoder(self,
                       tgt_vocab_size,
                       tgt_max_len,
                       embedding_dim,
                       decoder_num_layers,
                       decoder_dim,
                       number_of_head,
                       embedding_dropout,
                       decoder_dropout,
                       activation='relu',
                       shared_embedding=None
                       ):
        if shared_embedding is None:
            tgt_embedding_layer = EmbeddingLayer(tgt_vocab_size, embedding_dim)
            positional_encoder = PositionalEncoder(model_dim=embedding_dim,
                                                   dropout=embedding_dropout,
                                                   max_len=tgt_max_len * 2)
            positional_embedding = PositionalEmbedding(tgt_embedding_layer, positional_encoder)
        else:
            positional_embedding = shared_embedding
        decoder_layer = TransformerDecoderLayer(d_model=embedding_dim,
                                                nhead=number_of_head,
                                                dim_feedforward=decoder_dim,
                                                dropout=decoder_dropout,
                                                activation=activation)
        layer_norm = LayerNorm(embedding_dim)

        decoder = TransformerDecoder(decoder_layer, decoder_num_layers, layer_norm)
        return positional_embedding, decoder

    def create_model(self):

        src_positional_embedding, encoder = self.create_encoder(
            src_vocab_size=self.src_vocab_size,
            src_max_len=self.max_len,
            embedding_dim=self.embedding_dim,
            encoder_num_layers=self.encoder_num_layers,
            encoder_dim=self.encoder_dim,
            number_of_head=self.number_of_head,
            embedding_dropout=self.embedding_dropout,
            encoder_dropout=self.encoder_dropout,
            activation=self.activation)
        tgt_positional_embedding, decoder = self.create_decoder(
            tgt_vocab_size=self.tgt_vocab_size,
            tgt_max_len=self.max_len,
            embedding_dim=self.embedding_dim,
            decoder_num_layers=self.decoder_num_layers,
            decoder_dim=self.decoder_dim,
            number_of_head=self.number_of_head,
            embedding_dropout=self.embedding_dropout,
            decoder_dropout=self.decoder_dropout,
            activation=self.activation,
            shared_embedding=src_positional_embedding if self.share_embedding else None)

        if self.share_embedding:
            assert src_positional_embedding is tgt_positional_embedding

        generator = Generator(LinearLayer(self.embedding_dim,
                                          self.tgt_vocab_size,
                                          dropout=self.generator_dropout,
                                          bias=self.generator_bias))
        if self.share_embedding:
            generator.output_layer.linear_layer.weight = tgt_positional_embedding.embedding_layer.embedding.weight
        transformer = TransformerModel(
            src_embedding_layer=src_positional_embedding,
            transformer_encoder=encoder,
            transformer_decoder=decoder,
            tgt_embedding_layer=tgt_positional_embedding,
            generator_layer=generator)

        self.model = transformer
        return self.model

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
        last_decoder_layer = self.model.transformer_decoder.layers[-1]
        for name, module in last_decoder_layer.named_modules():
            print(name)
            if name == "multihead_attn":
                module.register_forward_hook(self.forward_hook)
        sys.stdout.flush()

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
                model.module.src_embedding_layer.embedding_layer.load_embeddings_from_numpy(self.source_embeddings_path,
                                                                                            self.embedding_word_dict_path,
                                                                                            dataset_manager.src_word_to_id_dict,
                                                                                            self.pad_token)
            else:
                model.src_embedding_layer.embedding_layer.load_embeddings_from_numpy(self.source_embeddings_path,
                                                                                     self.embedding_word_dict_path,
                                                                                     dataset_manager.src_word_to_id_dict,
                                                                                     self.pad_token)

    def loss_forward(self, current_epoch, i, packed_data, model, criterions=None, device=None):
        if not device:
            device = self.device

        if not criterions:
            criterions = self.criterions

        criterion = criterions[self.criterion]

        sos_src = packed_data[0].to(device)
        # eos_src = packed_data[1].to(device)
        sos_tgt = packed_data[2].to(device)
        eos_tgt = packed_data[3].to(device)
        src_len = packed_data[4].squeeze(1).to(device)
        tgt_len = packed_data[5].squeeze(1).to(device)

        src_padding_mask = generate_mask(src_len, device=device)
        tgt_padding_mask = generate_mask(tgt_len, device=device)
        predictions = model(sos_src.transpose(0, 1),
                            src_padding_mask.squeeze(-1),
                            sos_tgt.transpose(0, 1),
                            tgt_padding_mask.squeeze(-1))
        predictions = pack_padded_sequence(predictions.transpose(0, 1), tgt_len, True, False)
        e_txts = pack_padded_sequence(eos_tgt, tgt_len, True, False)

        num_of_tokens = float(e_txts.data.shape[0])
        num_of_sents = float(tgt_len.shape[0])
        loss = criterion(predictions.data, e_txts.data)
        loss = self.normalize_loss(loss, num_of_sents, num_of_tokens)

        loss = loss / self.optimize_delay
        batch_dict = {'loss':loss ,
                      "num_of_sents":num_of_sents, "num_of_tokens":num_of_tokens}
        return batch_dict

    @torch.no_grad()
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
        # eos_src = packed_data[1].to(device)
        # sos_tgt = packed_data[2].to(device)
        # eos_tgt = packed_data[3].to(device)
        src_len = packed_data[4].squeeze(1).to(device)
        # tgt_len = packed_data[5].squeeze(1).to(device)

        src_padding_mask = generate_mask(src_len, device=device).squeeze(-1)
        memory = model.encode(sos_src.transpose(0, 1), src_padding_mask)

        max_decode_step = int(max(src_len) * max_decode_step_ratio)
        # max_decode_step = min(max_decode_step, int(max(src_len) * max_decode_step_ratio))
        seqs = beam_search_decode_transformer(model,
                                              memory,
                                              src_padding_mask,
                                              tgt_word_to_id_dict,
                                              sos=tgt_sos,
                                              eos=eos,
                                              beam_width=beam_width,
                                              max_decode_step=max_decode_step)

        return seqs, None
