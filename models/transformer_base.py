import copy
import os
import numpy as np
import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from torch.nn import Dropout, Linear, LayerNorm, ModuleList, Module
from models.attention import MultiheadAttention


class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, num_heads, norm=None):
        """TransformerDecoder is a stack of N decoder layers

        Examples::
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        out = transformer_decoder(tgt, memory)

        :param decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        :param num_layers: the number of sub-decoder-layers in the decoder (required).
        :type num_layers: the layer normalization component (optional).

        """
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm = norm

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                get_attention_weights: Optional[str] = False) -> Tensor:
        """
        Pass the inputs (and mask) through the decoder layer in turn.
            :param tgt: the sequence to the decoder (required).
            :param memory: the sequence from the last layer of the encoder (required).
            :param tgt_mask: the mask for the tgt sequence (optional).
            :param memory_mask: the mask for the memory sequence (optional).
            :param tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            :param memory_key_padding_mask: the mask for the memory keys per batch (optional).
            :param get_attention_weights: get attention weights for each attention layer
        """
        output = tgt

        if get_attention_weights:
            # Dimensions [Bsize, Nlayers, Nheads, SeqDim, SeqDim]
            self_attention_weights = torch.zeros((output.shape[0],
                                                  self.num_layers,
                                                  self.num_heads,
                                                  output.shape[1],
                                                  output.shape[1]))

            # Dimensions [Bsize, Nlayers, Nheads, Seq1Dim, Seq2Dim]
            attention_weights = torch.zeros((output.shape[0],
                                             self.num_layers,
                                             self.num_heads,
                                             output.shape[1],
                                             memory.shape[1]))

            # Dimensions [Bsize, Nlayers, Nheads, Seq1Dim, Seq2Dim]
            saw_queries = torch.zeros((output.shape[0],
                                       self.num_layers,
                                       self.num_heads,
                                       output.shape[1],
                                       output.shape[2] // self.num_heads))
            saw_keys = torch.zeros_like(saw_queries)
            aw_queries = torch.zeros_like(saw_queries)
            aw_keys = torch.zeros((output.shape[0],
                                   self.num_layers,
                                   self.num_heads,
                                   memory.shape[1],
                                   memory.shape[2] // self.num_heads))

            # Iterate over all layers
            for idx, mod in enumerate(self.layers):
                output, saw, aw, saw_q, saw_k, aw_q, aw_k = mod(output,
                                                                memory,
                                                                tgt_mask=tgt_mask,
                                                                memory_mask=memory_mask,
                                                                tgt_key_padding_mask=tgt_key_padding_mask,
                                                                memory_key_padding_mask=memory_key_padding_mask,
                                                                get_attention_weights=get_attention_weights)
                self_attention_weights[:, idx] = saw
                attention_weights[:, idx] = aw
                saw_queries[:, idx] = saw_q.reshape(saw_queries.shape[0], saw_queries.shape[2], saw_queries.shape[3],
                                                    saw_queries.shape[4])
                saw_keys[:, idx] = saw_k.reshape(saw_keys.shape[0], saw_keys.shape[2], saw_keys.shape[3],
                                                 saw_keys.shape[4])
                aw_queries[:, idx] = aw_q.reshape(aw_queries.shape[0], aw_queries.shape[2], aw_queries.shape[3],
                                                  aw_queries.shape[4])
                aw_keys[:, idx] = aw_k.reshape(aw_keys.shape[0], aw_keys.shape[2], aw_keys.shape[3], aw_keys.shape[4])

            if self.norm is not None:
                output = self.norm(output)
            return output, self_attention_weights, attention_weights, saw_queries, saw_keys, aw_queries, aw_keys

        else:
            for mod in self.layers:
                output = mod(output,
                             memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             get_attention_weights=get_attention_weights)
            if self.norm is not None:
                output = self.norm(output)
            return output


class BertlikeDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, num_heads, norm=None):
        """TransformerDecoder is a stack of N decoder layers

        Examples::
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        out = transformer_decoder(tgt, memory)

        :param decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        :param num_layers: the number of sub-decoder-layers in the decoder (required).
        :type num_layers: the layer normalization component (optional).

        """
        super(BertlikeDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.norm = norm

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                get_attention_weights: Optional[str] = False) -> Tensor:
        """
        Pass the inputs (and mask) through the decoder layer in turn.
            :param tgt: the sequence to the decoder (required).
            :param memory: the sequence from the last layer of the encoder (required).
            :param tgt_mask: the mask for the tgt sequence (optional).
            :param memory_mask: the mask for the memory sequence (optional).
            :param tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            :param memory_key_padding_mask: the mask for the memory keys per batch (optional).
            :param get_attention_weights: get attention weights for each attention layer
        """
        # output = tgt

        if get_attention_weights:
            # raise NotImplementedError
            # Dimensions [Bsize, Nlayers, Nheads, SeqDim, SeqDim]
            tgt_self_attention_weights = torch.zeros((tgt.shape[0],
                                                  self.num_layers,
                                                  self.num_heads,
                                                  tgt.shape[1],
                                                  tgt.shape[1]+memory.shape[1]))

            memory_self_attention_weights = torch.zeros((memory.shape[0],
                                                  self.num_layers,
                                                  self.num_heads,
                                                  memory.shape[1],
                                                  tgt.shape[1]+memory.shape[1]))

            # # Dimensions [Bsize, Nlayers, Nheads, Seq1Dim, Seq2Dim]
            # attention_weights = torch.zeros((output.shape[0],
            #                                  self.num_layers,
            #                                  self.num_heads,
            #                                  tgt.shape[1],
            #                                  memory.shape[1]))

            # Dimensions [Bsize, Nlayers, Nheads, Seq1Dim, Seq2Dim]
            tgt_saw_queries = torch.zeros((tgt.shape[0],
                                       self.num_layers,
                                       self.num_heads,
                                       tgt.shape[1],
                                       tgt.shape[2] // self.num_heads))
            tgt_saw_keys = torch.zeros_like(tgt_saw_queries)
            memory_saw_queries = torch.zeros((memory.shape[0],
                                       self.num_layers,
                                       self.num_heads,
                                       memory.shape[1],
                                       memory.shape[2] // self.num_heads))
            memory_saw_keys = torch.zeros_like(memory_saw_queries)
            # aw_queries = torch.zeros_like(saw_queries)
            # aw_keys = torch.zeros((output.shape[0],
            #                        self.num_layers,
            #                        self.num_heads,
            #                        memory.shape[1],
            #                        memory.shape[2] // self.num_heads))

            # Iterate over all layers
            for idx, mod in enumerate(self.layers):
                tgt, memory, tgt_saw, tgt_saw_q, tgt_saw_k, memory_saw, memory_saw_q, memory_saw_k = mod(tgt,
                                                                memory,
                                                                tgt_mask=tgt_mask,
                                                                memory_mask=memory_mask,
                                                                tgt_key_padding_mask=tgt_key_padding_mask,
                                                                memory_key_padding_mask=memory_key_padding_mask,
                                                                get_attention_weights=get_attention_weights)
                tgt_self_attention_weights[:, idx] = tgt_saw
                memory_self_attention_weights[:, idx] = memory_saw
                # tgt_attention_weights[:, idx] = aw
                tgt_saw_queries[:, idx] = tgt_saw_q.reshape(tgt_saw_queries.shape[0], tgt_saw_queries.shape[2], tgt_saw_queries.shape[3],
                                                    tgt_saw_queries.shape[4])
                tgt_saw_keys[:, idx] = tgt_saw_k.reshape(tgt_saw_keys.shape[0], tgt_saw_keys.shape[2], tgt_saw_keys.shape[3],
                                                 tgt_saw_keys.shape[4])
                memory_saw_queries[:, idx] = memory_saw_q.reshape(memory_saw_queries.shape[0], memory_saw_queries.shape[2], memory_saw_queries.shape[3],
                                                    memory_saw_queries.shape[4])
                memory_saw_keys[:, idx] = memory_saw_k.reshape(memory_saw_keys.shape[0], memory_saw_keys.shape[2], memory_saw_keys.shape[3],
                                                 memory_saw_keys.shape[4])
                # aw_queries[:, idx] = aw_q.reshape(aw_queries.shape[0], aw_queries.shape[2], aw_queries.shape[3],
                #                                   aw_queries.shape[4])
                # aw_keys[:, idx] = aw_k.reshape(aw_keys.shape[0], aw_keys.shape[2], aw_keys.shape[3], aw_keys.shape[4])

            # if self.norm is not None:
            #     output = self.norm(output)
            if self.norm is not None:
                tgt = self.norm(tgt)
                memory = self.norm(memory)

            # return output, self_attention_weights, attention_weights, saw_queries, saw_keys, aw_queries, aw_keys
            return torch.cat([tgt, memory], axis=1), tgt_self_attention_weights, memory_self_attention_weights, \
                        tgt_saw_queries, tgt_saw_keys, memory_saw_queries, memory_saw_keys


        else:
            for mod in self.layers:
                tgt, memory = mod(tgt,
                             memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             get_attention_weights=get_attention_weights)
            if self.norm is not None:
                tgt = self.norm(tgt)
                memory = self.norm(memory)
            return torch.cat([tgt, memory], axis=1)

class BertlikeDecoderLayer(Module):
    __constants__ = ['batch_first']

    def __init__(self,
                 d_model,
                 n_heads,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 layer_norm_eps=1e-5,
                 batch_first=False,
                 device=None,
                 dtype=None) -> None:
        """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
        Args:
            :param d_model: the number of expected features in the input (required).
            :param n_heads: the number of heads in the multiheadattention models (required).
            :param dim_feedforward: the dimension of the feedforward network model (default=2048).
            :param dropout: the dropout value (default=0.1).
            :param activation: the activation function of intermediate layer, relu or gelu (default=relu).
            :param layer_norm_eps: the eps value in layer normalization components (default=1e-5).
            :param batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``False``.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BertlikeDecoderLayer, self).__init__()
        self.self_attn1 = MultiheadAttention(d_model,
                                            n_heads,
                                            dropout=dropout,
                                            batch_first=batch_first,
                                            **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(BertlikeDecoderLayer, self).__setstate__(state)


    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                get_attention_weights: Optional[str] = False) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            :param: tgt: the sequence to the decoder layer (required).
            :param: memory: the sequence from the last layer of the encoder (required).
            :param: tgt_mask: the mask for the tgt sequence (optional).
            :param: memory_mask: the mask for the memory sequence (optional).
            :param: tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            :param: memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """

        tgt_len = tgt.shape[1]
        memory_len = memory.shape[1]
        
        concat_seq = torch.cat([tgt, memory], axis=1)
        
        concat_seq2, self_attention_weights, self_attention_queries, self_attention_keys = self.self_attn1(concat_seq,
                                                                                                   concat_seq,
                                                                                                   concat_seq,
                                                                                                   attn_mask=tgt_mask,
                                                                                                   key_padding_mask=tgt_key_padding_mask)
    
        concat_seq = concat_seq + self.dropout2(concat_seq2)
        concat_seq = self.norm2(concat_seq)
        concat_seq2 = self.linear2(self.dropout(self.activation(self.linear1(concat_seq))))
        concat_seq = concat_seq + self.dropout3(concat_seq2)
        concat_seq = self.norm3(concat_seq)
        
        if get_attention_weights:
            # return concat_seq[:,:tgt_len], concat_seq[:,tgt_len:], self_attention_weights, self_attention_queries, self_attention_keys
            return concat_seq[:,:tgt_len], concat_seq[:,tgt_len:], \
                    self_attention_weights[:,:,:tgt_len], self_attention_queries[:,:tgt_len], self_attention_keys[:,:tgt_len], \
                    self_attention_weights[:,:,tgt_len:], self_attention_queries[:,tgt_len:], self_attention_keys[:,tgt_len:]
                    # outputs peptide-lengthed self-attention weights, HLA-lengthed self-attention weights
        return concat_seq[:,:tgt_len], concat_seq[:,tgt_len:]

class TransformerDecoderLayer(Module):
    __constants__ = ['batch_first']

    def __init__(self,
                 d_model,
                 n_heads,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 layer_norm_eps=1e-5,
                 batch_first=False,
                 device=None,
                 dtype=None) -> None:
        """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
        Args:
            :param d_model: the number of expected features in the input (required).
            :param n_heads: the number of heads in the multiheadattention models (required).
            :param dim_feedforward: the dimension of the feedforward network model (default=2048).
            :param dropout: the dropout value (default=0.1).
            :param activation: the activation function of intermediate layer, relu or gelu (default=relu).
            :param layer_norm_eps: the eps value in layer normalization components (default=1e-5).
            :param batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``False``.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model,
                                            n_heads,
                                            dropout=dropout,
                                            batch_first=batch_first,
                                            **factory_kwargs)

        self.multihead_attn = MultiheadAttention(d_model,
                                                 n_heads,
                                                 dropout=dropout,
                                                 batch_first=batch_first,
                                                 **factory_kwargs)
                                                 
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)


    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                get_attention_weights: Optional[str] = False) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            :param: tgt: the sequence to the decoder layer (required).
            :param: memory: the sequence from the last layer of the encoder (required).
            :param: tgt_mask: the mask for the tgt sequence (optional).
            :param: memory_mask: the mask for the memory sequence (optional).
            :param: tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            :param: memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        tgt2, self_attention_weights, self_attention_queries, self_attention_keys = self.self_attn(tgt,
                                                                                                   tgt,
                                                                                                   tgt,
                                                                                                   attn_mask=tgt_mask,
                                                                                                   key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
    
        tgt = self.norm1(tgt)
        
        [tgt2, attention_weights, attention_queries, attention_keys] = self.multihead_attn(tgt,
                                                                                           memory,
                                                                                           memory,
                                                                                           attn_mask=memory_mask,
                                                                                           key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if get_attention_weights:
            return tgt, self_attention_weights, attention_weights, self_attention_queries, self_attention_keys, attention_queries, attention_keys
        return tgt

def _get_clones(module, N):
    """
    Generate N deep copies of a torch layer.
    :param module:
    :param N:
    :return:
    """
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
