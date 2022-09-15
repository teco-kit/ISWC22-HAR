
# transformer的一层分为两个部分， 第一部分是selfattention，第二个部分是feedforward
# 以下代码首先编写self attention
# self attention 

import torch
import torch.nn as nn
import math
import numpy as np
from models.crossatten.attention_masks import Mask_dict
from models.crossatten.utils import DW_PW_projection
# ------------------------------------------------------------------
from einops import rearrange, repeat

####################         Mask Attention      ###############################
class MaskAttention(nn.Module):
    def __init__(self, 
                 mask_flag=True, 
                 mask_typ = "Triangular",
                 attention_dropout=0.1, 
                 output_attention=False):
        """
        mask_flag ： 是否使用mask，如果不使用，那么就是全局mask
        mask_typ  ： 如果使用mask，哪种？
        attention_dropout ： attention之后 score的dropout
        output_attention  ： bool，是否输出attentionmap
        """
        super(MaskAttention, self).__init__()
        self.mask_typ         = mask_typ
        #print("self.mask_typ ", self.mask_typ)
        self.mask_flag        = mask_flag
        self.output_attention = output_attention
        self.attn_drop        = nn.Dropout(attention_dropout)
				
    def forward(self, queries, keys, values):
        """
        queries : [Batch, Length, Heads, E]
        keys    : [Batch, Length, Heads, E]
        values  : [Batch, Length, Heads, D]

        返回的是两个东西
        1.  attn_values : 新的value  格式依旧是 [Batch, Length, Heads, D]
        2.  attention 的map
        """
        B, L, H, E = queries.shape
        _, _, _, D = values.shape
 


        queries = queries.permute(0, 2, 1, 3)                                                     # [batch, heads, length, chanell]
        keys    = keys.permute(0, 2, 3, 1)                                                        # [batch, heads, chanell, length]
        attn    = torch.matmul(queries, keys)                                                     
        scale   =  1./math.sqrt(E) 
        attn    = scale * attn
        
        if self.mask_flag:
            attn_mask = Mask_dict[self.mask_typ](B, L, device=queries.device)
            attn.masked_fill_(attn_mask.mask, -np.inf)                                       #其实就是把不想要的地方设为True，然后再在这些地方加上 -inf        


        attn = self.attn_drop(torch.softmax(attn , dim=-1))
        
        values = values.permute(0, 2, 1, 3)                                                       # [batch, heads, length, chanell]
        attn_values = torch.matmul(attn, values).permute(0,2,1,3)                                 # [batch, length, heads, chanell]

        
        if self.output_attention:
            return (attn_values.contiguous(), attn)
        else:
            return (attn_values.contiguous(), None)


####################         Attention Layer      ###############################
class AttentionLayer(nn.Module):
    def __init__(self, 
                 attention, 
                 input_dim,
                 d_model, 
                 n_heads, 
                 d_keys               =  None, 
                 d_values             =  None, 
                 causal_kernel_size   =  3, 
                 value_kernel_size    =  1,
                 bias                 = False,
                 padding_mode         = 'replicate',
                 projection_dropout   =  0.1,
                 light_weight         = False):
        """

        attention          :    要进行什么样子的attention？Probmask？seasonal？还是全局的？ 默认就是full吧
        d_model            :    输入的维度
        n_heads            :    注意力的个数
        d_keys             ：    query和key的映射维度 ，默认是和d_model一样大
        d_values           ：    value的映射维度，默认是和d_model一样大
        causal_kernel_size :    是否通过local conv进行提取特征。 如果等于1， 就是linear. 如果大于1，就是1d conv
        value_kernel_size  :    和上面参数一致
        attention_dropout  ：    
        
	    """

        super(AttentionLayer, self).__init__()

        self.n_heads = n_heads
        self.d_keys = d_keys or d_model                                                              # 每个head中，key和query的维度
        self.d_values = d_values or d_model                                                          # 每个head中，value 的维度, 一般情况应该和key一样

        # 因为是时间序列，这里采取的causal attention，通过设置kernel的大小，可以是linear
        # 提取key和query的kernel大小，提取value的kernel大小, 当等于1时，就是linear，当大于1时就是conv
        self.causal_kernel_size = causal_kernel_size
        self.value_kernel_size  = value_kernel_size 

        self.projection_dropout = projection_dropout

        # 初始化4个projection，分别时key，query， value以及最后新value的out的projection
        if light_weight:
            self.query_projection = DW_PW_projection(c_in         = input_dim, 
                                                     c_out        = self.d_keys, 
                                                     kernel_size  = self.causal_kernel_size,
                                                     bias         = bias, 
                                                     padding_mode = padding_mode)

            self.key_projection = DW_PW_projection(c_in         = input_dim, 
                                                   c_out        = self.d_keys, 
                                                   kernel_size  = self.causal_kernel_size,
                                                   bias         = bias, 
                                                   padding_mode = padding_mode)

            self.value_projection = DW_PW_projection(c_in         = input_dim, 
                                                     c_out        = self.d_values, 
                                                     kernel_size  = self.value_kernel_size,
                                                     bias         = bias, 
                                                     padding_mode = padding_mode)
            # 与前三个projection的输入维度不一样，因为这里的输入时attention后的新value
            # 由于有skip的机制，所以整个attention的输入和输出要保持一直
            self.out_projection = DW_PW_projection(c_in         = self.d_values, 
                                                   c_out        = d_model, 
                                                   kernel_size  = self.value_kernel_size,
                                                   bias         = bias, 
                                                   padding_mode = padding_mode)

        else:
            self.query_projection = nn.Conv1d(in_channels  = input_dim,
                                              out_channels = self.d_keys, 
                                              kernel_size  = self.causal_kernel_size,
                                              padding      =  int(self.causal_kernel_size/2),
                                              bias         =  bias,  
                                              padding_mode = padding_mode)

            self.key_projection = nn.Conv1d(in_channels  = input_dim,
                                            out_channels = self.d_keys, 
                                            kernel_size  = self.causal_kernel_size,
                                            padding      =  int(self.causal_kernel_size/2),
                                            bias         =  bias,  
                                            padding_mode = padding_mode)

            self.value_projection = nn.Conv1d(in_channels  = input_dim,
                                              out_channels = self.d_values , 
                                              kernel_size  = self.value_kernel_size,
                                              padding      =  int(self.value_kernel_size/2),
                                              bias         =  bias,  
                                              padding_mode = padding_mode)

            # 与前三个projection的输入维度不一样，因为这里的输入时attention后的新value
            # 由于有skip的机制，所以整个attention的输入和输出要保持一直
            self.out_projection = nn.Conv1d(in_channels  = self.d_values ,
                                            out_channels = d_model,
                                            kernel_size  = self.value_kernel_size,
                                            padding      = int(self.value_kernel_size/2),
                                            bias         = bias,  
                                            padding_mode = padding_mode)

        self.inner_attention = attention



        self.proj_drop = nn.Dropout(projection_dropout)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv1d):
        #        nn.init.kaiming_normal_(m.weight)


    def forward(self, queries, keys, values):

        B, L_Q, I_Q = queries.shape
        _, L_K, I_K = keys.shape
        _, L_V, I_V = values.shape                                                                   # 理论上所有的L_和I_是一模一样的
        H = self.n_heads

        # # 以上 B L C 中的C是包含了所有Head的特征，映射之后拆分为，每个head的特征，也就是， [B, L, H, C] 
        #  ========================== value projection ==========================
        values               = self.value_projection(values.permute(0, 2, 1)).permute(0, 2, 1)
        values               = values.view(B, L_V, H, -1)

        # ========================== query  keys projection ==========================
        queries              = self.query_projection(queries.permute(0, 2, 1)).permute(0, 2, 1)
        queries              = queries.view(B, L_Q, H, -1)

        keys                 = self.key_projection(keys.permute(0, 2, 1)).permute(0, 2, 1)
        keys                 = keys.view(B, L_K, H, -1)   


        # ========================== attention ==========================
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )
        #out = out.view(B, L_V, -1)                                                                 # TODO L_V?                                                 
        out = rearrange(out, 'b l h c -> b l (h c)')
        # ========================== Out Projection ==========================

        out                 = self.out_projection(out.permute(0, 2, 1)).permute(0, 2, 1)
			
        out                 = self.proj_drop(out)
        return out, attn



class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, attention):
        super().__init__()

        self.attention = attention

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, query, context):

        query = self.project_in(query)

        context = torch.cat((context,query), dim = 1)

        value_query, _ = self.attention(query, context, context)
        value_query = self.project_out(value_query)
        return value_query


class CrossAttentionLayer(nn.Module):
    def __init__(self, args, depth):

        super(CrossAttentionLayer, self).__init__()

        self.layers = nn.ModuleList([])

        ts_dim = args.token_d_model
        fq_dim = args.token_d_model


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(ts_dim, fq_dim, AttentionLayer(attention          = MaskAttention(mask_flag          = True, 
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
                                                            light_weight       = args.light_weight)),
                nn.LayerNorm(ts_dim),

                ProjectInOut(fq_dim, ts_dim, AttentionLayer(attention          = MaskAttention(mask_flag          = True, 
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
                                                            light_weight       = args.light_weight)),
                nn.LayerNorm(fq_dim),
            ]))

    def forward(self, ts_tokens, fq_tokens):
        """
        ts_tokens : B, L+1, C
        fq_tokens : B, L+1, C
        """
        (ts_cls, ts_patch_tokens), (fq_cls, fq_patch_tokens) = map(lambda t: (t[:, -1:], t[:, :-1]), (ts_tokens, fq_tokens))

        for ts_attend_fq, ts_layernorm, fq_attend_ts, fq_layernorm in self.layers:

            ts_cls = ts_attend_fq(query = ts_cls, context = fq_patch_tokens) + ts_cls
            ts_cls = ts_layernorm(ts_cls)

            fq_cls = fq_attend_ts(query = fq_cls, context = ts_patch_tokens) + fq_cls
            fq_cls = fq_layernorm(fq_cls)

        ts_tokens = torch.cat((ts_patch_tokens, ts_cls), dim = 1)
        fq_tokens = torch.cat((fq_patch_tokens, fq_cls), dim = 1)

        return ts_tokens, fq_tokens
