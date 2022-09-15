import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.crossatten.embedding import TimeEmbedder,FreqEmbedder
from models.crossatten.encoder import EncoderLayer, ConvLayer, Encoder
from models.crossatten.attention import AttentionLayer, MaskAttention, CrossAttentionLayer



class CrossTransformer(nn.Module):
    def __init__(self, args , depth  ):
        super(CrossTransformer, self).__init__()


        """
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
        """
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TSCtransformer(args, args.t_depth),
                TSCtransformer(args, args.t_depth),
                CrossAttentionLayer(args, args.cross_atten_depth)
            ]))

    def forward(self, ts_tokens, fq_tokens):
        for ts_enc, fq_enc, cross_attend in self.layers:
            ts_tokens, _ = ts_enc(ts_tokens)
            fq_tokens, _ = fq_enc(fq_tokens)
            ts_tokens, fq_tokens = cross_attend(ts_tokens, fq_tokens)

        return ts_tokens, fq_tokens


class TSCtransformer(nn.Module):
    def __init__(self, args, depth):
        super(TSCtransformer, self).__init__()
        # all parameters are saved in self.args
        self.args = args
        # ================================ Encoding part ================================

        attention_list = []
        for l in range(depth):
  
            attention_list.append(AttentionLayer(attention          = MaskAttention(mask_flag          = True, 
                                                                                    mask_typ           = args.attention_layer_types,
                                                                                    attention_dropout  = args.attention_dropout, 
                                                                                    output_attention   = args.output_attention ),
                                                 input_dim          = args.token_d_model,
                                                 d_model            = args.token_d_model, 
                                                 n_heads            = args.n_heads,
                                                 d_keys             = args.d_keys, 
                                                 d_values           = args.d_values, 
                                                 causal_kernel_size = args.causal_kernel_size, 
                                                 value_kernel_size  = args.value_kernel_size,
                                                 bias               = args.bias,
                                                 padding_mode       = args.padding_mode,
                                                 projection_dropout = args.projection_dropout,
                                                 light_weight       = args.light_weight))



        self.encoder = Encoder([EncoderLayer(attention           = attention_list[l],
                                             d_model             = args.token_d_model,
                                             dim_feedforward     = args.feedforward_dim, 
                                             feedforward_dropout = args.feedforward_dropout,
                                             activation          = args.feedforward_activation,
                                             norm_type           = args.feedforward_norm_type,
                                             forward_kernel_size = args.forward_kernel_size,
                                             bias                = args.bias,
                                             padding_mode        = args.padding_mode,
                                             light_weight        = args.light_weight) for l in range(depth)],

                               [ConvLayer( c_in            = args.token_d_model, 
                                           c_out           = args.token_d_model,
                                           bias            = args.bias,
                                           padding_mode    = args.padding_mode,
                                           conv_norm       = args.conv_norm, 
                                           conv_activation = args.conv_activation,
                                           light_weight        = args.light_weight) for l in range(depth-1)] if args.distil else None
                               )
        print("build encoder Done")




    def forward(self, x):

        x, attns = self.encoder(x)

        return x, attns

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            print("init linear")
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            print("init LayerNorm")
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            print("init Conv1d")
            nn.init.kaiming_normal_(m.weight)

    @staticmethod
    def sinusoidal_embedding(length, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(length)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)





class Cross_TS(nn.Module):
    def __init__(self, args):
        super(Cross_TS, self).__init__()

        self.args = args
 
        print("beginn to build model")
        # ================================ Embedding part ================================
 
        if self.args.model_type == "time":
            self.time_value_embedding = TimeEmbedder(args)

        elif self.args.model_type == "freq":
            self.freq_value_embedding = FreqEmbedder(args)

        else: # self.args.model_type == "cross":
            self.time_value_embedding = TimeEmbedder(args)
            self.freq_value_embedding = FreqEmbedder(args)
        print("build embedding Done")






        # ================================ Transformer part ================================
        if self.args.model_type == "cross":
            self.cross_transformer = CrossTransformer(args, args.cross_depth)
            self.ts_transformer    = TSCtransformer(args, 1)
            self.fq_transformer    = TSCtransformer(args, 1)
        else:
            self.transformer = TSCtransformer(args, args.single_depth)
        print("build transformer Done")






        # ================================ Prediction part ================================
        if self.args.model_type == "time":
            self.time_classes_prediction = nn.Linear(args.token_d_model, args.num_classes)

        elif self.args.model_type == "freq":
            self.freq_classes_prediction = nn.Linear(args.token_d_model, args.num_classes)

        else: # self.args.model_type == "cross":
            self.time_classes_prediction = nn.Linear(args.token_d_model, args.num_classes)
            self.freq_classes_prediction = nn.Linear(args.token_d_model, args.num_classes)
	
        print("build prediction Done")

    def time_forward(self, x_ts):

        x_ts = self.time_value_embedding(x_ts)
        x_ts, attns = self.transformer(x_ts)
        ts_cls = x_ts[:,-1]
        ts_cls = self.time_classes_prediction(ts_cls)

        return ts_cls, attns

    def freq_forward(self, x_fq):

        x_fq = self.freq_value_embedding(x_fq)
        x_fq, attns = self.transformer(x_fq)
        fq_cls = x_fq[:,-1]
        fq_cls = self.freq_classes_prediction(fq_cls)

        return fq_cls, attns

    def cross_forward(self, x_ts, x_fq):
        x_ts = self.time_value_embedding(x_ts)
        x_fq = self.freq_value_embedding(x_fq)

        attns = None
        x_ts, x_fq = self.cross_transformer(x_ts, x_fq)

        x_ts, attns = self.ts_transformer(x_ts)
        x_fq, attns = self.fq_transformer(x_fq)
		
        ts_cls, fq_cls = map(lambda t: t[:, -1], (x_ts, x_fq))

        ts_cls = self.time_classes_prediction(ts_cls)
        fq_cls = self.freq_classes_prediction(fq_cls)

        return ts_cls+fq_cls, attns

    def forward(self, x1, x2=None):
        model_type = self.args.model_type
        if model_type == "cross":
            assert torch.is_tensor(x1)
            assert torch.is_tensor(x2)

            cls, attns = self.cross_forward(x1, x2)

        elif model_type == "time":
            assert torch.is_tensor(x1)
            cls, attns = self.time_forward(x1)


        else: # model_type == "freq":
            assert torch.is_tensor(x1)
            cls, attns = self.freq_forward(x1)



        return cls, attns

# ========================================= Basic Version ==========================================

from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
import math
from torch import nn, Tensor
from typing import Optional, Any

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))



class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src

# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
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
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))



class TSTransformer_Basic(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, args):
        super(TSTransformer_Basic, self).__init__()
        self.args = args

        print("beginn to build model")
        # ================================ Embedding part ================================
        self.project_inp = nn.Linear(args.c_in, args.token_d_model)
        self.pos_enc     = get_pos_encoder(args.positional_embedding)(args.token_d_model, dropout=args.input_embedding_dropout, max_len=args.input_length)

        # ================================ Encoder part ================================
        if args.norm_type == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(args.token_d_model, args.n_heads, args.dim_feedforward, args.attn_dropout, activation=args.activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(args.token_d_model, args.n_heads, args.dim_feedforward, args.attn_dropout, activation=args.activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, args.num_layers)
        self.act = _get_activation_fn(args.activation)
        self.dropout1 = nn.Dropout(args.attn_dropout)
        # ================================ Prediction part ================================
  
        self.num_classes = args.num_classes
        self.output_layer = self.build_output_module(args.token_d_model, args.input_length, args.num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        return output_layer

    def forward(self, x):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        # ================================ Embedding part ================================
        x = x.permute(1, 0, 2)
        x = self.project_inp(x) * math.sqrt(self.args.token_d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        x = self.pos_enc(x)  # add positional encoding

        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        x = self.transformer_encoder(x)  # (seq_length, batch_size, d_model)
        x = self.act(x)  # the output transformer encoder/decoder embeddings don't include non-linearity
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        x = self.dropout1(x)

        # Output
        # x = x * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        x = x.reshape(x.shape[0], -1)  # (batch_size, seq_length * d_model)
        x = self.output_layer(x)  # (batch_size, num_classes)

        return x

		