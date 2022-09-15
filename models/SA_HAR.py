import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ====================================  model ==================================== #
class ConvBlock(nn.Module):
    """
    Normal convolution block
    """
    def __init__(self, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super(ConvBlock, self).__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1),padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, 1, (self.filter_width, 1), dilation=(self.dilation, 1), stride=(1,1),padding='same')
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm1(out)

        out = self.conv2(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm2(out)

        return out

class SensorAttention(nn.Module):
    def __init__(self, input_shape, nb_filters ):
        super(SensorAttention, self).__init__()
        self.ln = nn.LayerNorm(input_shape[3])        #  channel的维度
        
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=nb_filters, kernel_size=3, dilation=2, padding='same')
        self.conv_f = nn.Conv2d(in_channels=nb_filters, out_channels=1, kernel_size=1, padding='same')
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=3)

        
    def forward(self, inputs):
        '''
        input: [batch  * length * channel]
        output: [batch, 1, length, d]
        '''
        # layer norm 在最后一个维度，  原文是在feature上
        inputs = self.ln(inputs)               
        # 增加 维度， tensorflow 是在最后一个维度上加， torch是第一个
        x = inputs.unsqueeze(1)                
        # b 1 L C
        x = self.conv_1(x)              
        x = self.relu(x)  
        # b 128 L C
        x = self.conv_f(x)               
        # b 1 L C
        x = self.softmax(x)
        x = x.squeeze(1)                  # batch * channel * len 
        # B L C
        return torch.mul(inputs, x), x    # batch * channel * len, batch * channel * len 


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(AttentionLayer, self).__init__()


        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection   = nn.Linear(d_model, d_model, bias=True)
        self.value_projection = nn.Linear(d_model, d_model, bias=True)
        self.out_projection   = nn.Linear(d_model, d_model, bias=True)

        self.n_heads = n_heads


    def forward(self, queries, keys, values):
        B, L, _ = queries.shape

        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)



        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        _, _, _, E = queries.shape
        scale = 1./math.sqrt(E)
        Attn = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", Attn, values).contiguous()

        out = V.view(B, L, -1)
        out = self.out_projection(out)
        return out, Attn
    

class EncoderLayer(nn.Module):
    def __init__(self,  d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()



        self.attention = AttentionLayer(d_model, n_heads)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)    
        
        
        d_ff = d_ff or 4*d_model
        self.ffn1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear(d_ff, d_model, bias=True)

         
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)               
        

        self.dropout2 = nn.Dropout(p=dropout)



    def forward(self, x):

        attn_output, attn = self.attention( x, x, x )
        attn_output = self.dropout1(attn_output)
        out1  = self.layernorm1(x + attn_output)

        ffn_output = self.ffn2(self.relu(self.ffn1(out1)))
        ffn_output = self.dropout2(ffn_output)
        out2 =  self.layernorm2(out1 + ffn_output)

        return out2

class AttentionWithContext(nn.Module):
    def __init__(self, token_d_model):
        super(AttentionWithContext, self).__init__()
        self.W = nn.Linear(token_d_model, token_d_model)
        self.tanh = nn.Tanh()
        self.u = nn.Linear(token_d_model, 1, bias=False)
    def forward(self, inputs):
        uit = self.W(inputs)
        uit = self.tanh(uit)
        ait = self.u(uit)
        outputs = torch.matmul(F.softmax(ait, dim=1).transpose(-1, -2),inputs).squeeze(-2)
        return outputs


class SA_HAR(nn.Module):
    def __init__(self, 
                 input_shape, 
                 nb_classes, 
                 filter_scaling_factor, 
                 config):
        super(SA_HAR, self).__init__()

        self.nb_filters     = int(filter_scaling_factor*config["nb_filters"])

        self.first_conv = ConvBlock(filter_width=5, 
                                    input_filters=input_shape[1], 
                                    nb_filters=self.nb_filters, 
                                    dilation=1, 
                                    batch_norm=True).double()
        
        self.SensorAttention = SensorAttention(input_shape,self.nb_filters)
        self.conv1d = nn.Conv1d(in_channels=input_shape[3], out_channels=self.nb_filters, kernel_size=1)
        
        
        #self.pos_embedding = nn.Parameter(self.sinusoidal_embedding(input_shape[2], self.nb_filters), requires_grad=False)
        #self.pos_dropout = nn.Dropout(p=0.2) 
        
        self.EncoderLayer1 = EncoderLayer( d_model = self.nb_filters, n_heads =4 , d_ff = self.nb_filters*4)
        self.EncoderLayer2 = EncoderLayer( d_model = self.nb_filters, n_heads =4 , d_ff = self.nb_filters*4)


        self.AttentionWithContext = AttentionWithContext(self.nb_filters)

        self.fc1 = nn.Linear(self.nb_filters, 4*nb_classes)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.2)

        self.fc_out = nn.Linear(4*nb_classes, nb_classes)      # 从d_dim到6classes, 取72是论文中说的4倍classes数（4*18）

    
    def forward(self,x): 
        # x -- > B  fin  length Chennel
        x = self.first_conv(x)
        x = x.squeeze(1) 
        # x -- > B length Chennel
	
        # B L C
        si, _ = self.SensorAttention(x) 
        
        # B L C
        x = self.conv1d(si.permute(0,2,1)).permute(0,2,1) 
        x = self.relu(x)            
        # B L C
        #x = x + self.pos_embedding
        #x = self.pos_dropout(x)

        x = self.EncoderLayer1(x)            # batch * len * d_dim
        x = self.EncoderLayer2(x)            # batch * len * d_dim
        
        # Global Temporal Attention
        x = self.AttentionWithContext(x)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc_out(x)
        
        return x
    
    @staticmethod
    def sinusoidal_embedding(length, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(length)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
