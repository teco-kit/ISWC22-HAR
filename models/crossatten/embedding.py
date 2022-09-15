import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
from models.crossatten.utils import DW_PW_projection,  Norm_dict, Activation_dict
# TODO 所有循环结构应该呈现灵活性，每一层都不能一样！

div_sa = 3



class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """
    Create and initialize a `nn.Conv1d` layer with spectral normalization.
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    # return spectral_norm(conv)
    return conv

class SelfAttention(nn.Module):
    """
    # self-attention implementation from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    Self attention layer for nd
    """
    def __init__(self, n_channels: int, div):
        super(SelfAttention, self).__init__()
        self.div = div
        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels//div)
            self.key = conv1d(n_channels, n_channels//div)
            if self.div > 1:
                self.value = conv1d(n_channels, n_channels//div)
                self.value1 = conv1d(n_channels, n_channels//div)
            else:
                self.value = conv1d(n_channels, n_channels)
                self.value1 = None
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
            self.value = conv1d(n_channels, n_channels)
            self.value1 = None
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        batch , filter, Channel, _ = x.shape # _ = 1
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        if self.value1 is not None:
            x = self.value1(x)
        o = self.gamma * torch.bmm(h, beta) + x
        #return o.view(batch,int(filter/3),Channel,1).contiguous()
        return o.view(batch,int(filter/self.div),Channel,1).contiguous()
# ---------------------- PositionalEmbedding ----------------------

class PositionalEmbedding(nn.Module):
    """
    input shape should be (batch, seq_length, feature_channel)
    
    """
    def __init__(self, pos_d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        
        
        pe = torch.zeros(max_len, pos_d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, pos_d_model, 2).float() * -(math.log(10000.0) / pos_d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)# [1, max_len, pos_d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)] # select the the length same as input


    def vis_pos_heat(self, length):
        heat = self.pe[:, :length]
        plt.figure(figsize=(15,5))
        sns.heatmap(heat.detach().numpy()[0], linewidth=0)
        plt.ylabel("length")
        plt.xlabel("embedding")





# ---------------------- Time Series input Embdding ----------------------

class Forward_block(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size,
                 stride                 =  1, 
                 conv_bias              = False,
                 activation             = "relu",
                 norm_type              = "batch",
                 max_pool               = False,
                 pooling_kernel_size    = 3, 
                 pooling_stride         = 2,
                 pooling_padding        = 1,
                 padding_mode           = 'replicate',
                 light_weight           = False):
        """
        embedding的block 由 conv --> norm --> activation --> maxpooling组成
        """
        super(Forward_block, self).__init__() 
        if light_weight:
            self.conv = DW_PW_projection(c_in         = c_in, 
                                         c_out        = c_out,
                                         kernel_size  = kernel_size,
                                         stride       =  stride,
                                         bias         = conv_bias, 
                                         padding_mode = padding_mode)
        else:
            self.conv = nn.Conv1d(in_channels  =  c_in, 
                                  out_channels =  c_out,
                                  kernel_size  =  kernel_size,
                                  padding      =  int(kernel_size/2),
                                  stride       =  stride,
                                  bias         =  conv_bias,
                                  padding_mode =  padding_mode)
        self.norm_type   = norm_type
        self.norm        = Norm_dict[norm_type](c_out)
        self.activation  = Activation_dict[activation]()
        self.max_pool    = max_pool
        if max_pool:
           self.maxpooling =  nn.MaxPool1d(kernel_size = pooling_kernel_size,
                                           stride      = pooling_stride,
                                           padding     = pooling_padding)
    def forward(self, x):

        x  = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.norm_type == "layer":
            x = self.activation(self.norm(x))
        else :
            x = self.activation(self.norm(x.permute(0, 2, 1)).permute(0, 2, 1))

        if self.max_pool:
            x = self.maxpooling(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class TokenEmbedding(nn.Module):
    def __init__(self,
                 c_in, 
                 token_d_model,
                 kernel_size            = 3, 
                 stride                 = 1, 
                 conv_bias              = False,
                 activation             = "relu",
                 norm_type              = "batch",
                 n_conv_layers          = 1,
                 in_planes              = None,
                 max_pool               = False,
                 pooling_kernel_size    = 3, 
                 pooling_stride         = 2,
                 pooling_padding        = 1,
                 padding_mode           = 'replicate',
                 light_weight           = False):
        """
        c_in  : 模型输入的维度
        token_d_model ： embedding的维度  TODO看看后面是需要被相加还是被cat
        kernel_size   : 每一层conv的kernel大小
    
        """
        super(TokenEmbedding, self).__init__()
        in_planes = in_planes or int(token_d_model/2)
        n_filter_list = [c_in] + [in_planes for _ in range(n_conv_layers - 1)] + [token_d_model]
        padding = int(kernel_size/2)


        self.conv_layers = []
        for i in range(n_conv_layers):
            self.conv_layers.append(Forward_block(c_in                = n_filter_list[i],
                                                  c_out               = n_filter_list[i + 1], 
                                                  kernel_size         = kernel_size,
                                                  stride              = stride, 
                                                  conv_bias           = conv_bias,
                                                  activation          = activation,
                                                  norm_type           = norm_type,
                                                  max_pool            = max_pool,
                                                  pooling_kernel_size = pooling_kernel_size, 
                                                  pooling_stride      = pooling_stride,
                                                  pooling_padding     = pooling_padding,
                                                  padding_mode        = padding_mode,
                                                  light_weight        = light_weight))

        self.conv_layers = nn.ModuleList(self.conv_layers)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv1d):
        #        nn.init.kaiming_normal_(m.weight)



    def forward(self, x):

        for layer in self.conv_layers:
            x = layer(x)
        return x

    def sequence_length(self, length=100, n_channels=3):
        return self.forward(torch.zeros((1, length,n_channels))).shape[1]

class Time_FeatureExtractor_attention(nn.Module):
    def __init__(self, input_shape, number_filter, token_d_model, conv_number=4, filter_size=5, activation = "ReLU"):
        super(Time_FeatureExtractor_attention, self).__init__()
        # input_shape: B L C
        print("Time_FeatureExtractor_attention with ", div_sa)
        c_in = input_shape[2]

        layers = []
        for i in range(conv_number):
            if i == 0:
                temp = 1
            else:
                temp = number_filter
            layers.append(nn.Sequential(
                nn.Conv2d(temp, number_filter, (filter_size, 1), padding ="same"),
                nn.ReLU(inplace=True),
                #nn.BatchNorm2d(number_filter)
            ))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(0.2)

        self.sa = SelfAttention(number_filter*div_sa, div=div_sa)
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()
        self.fc = nn.Linear(number_filter*c_in, token_d_model)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x shape batch 1  Length Channel
        for layer in self.layers:
            x = layer(x)
        # x shape batch number_filter, Length Channel
        batch, filter, length, channel = x.shape


        #refined = torch.cat(
        #    [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
        #    dim=-1,
        #)
        #x = refined.permute(0, 3, 1, 2)
        #x = x.reshape(x.shape[0], x.shape[1], -1)
        #x = self.dropout(x)
        #x = self.activation(self.fc(x))
        refined = [] 
        step = int(np.ceil(length/div_sa))

        for index in range(step):
            if index<step-1:
                temp = torch.unsqueeze(x[:, :, index*div_sa:(index+1)*div_sa, :].reshape(batch, -1, channel).contiguous(), dim=3)
            else:
                temp = torch.unsqueeze(x[:, :, -div_sa:, :].reshape(batch, -1, channel).contiguous(), dim=3)
            refined.append(self.sa(temp))

        refined = torch.cat(refined,   dim=-1)
        # refined shape batch numberfilter, C, Length/div_sa
        x = self.activation(refined.permute(0, 3, 1, 2))
        # x shape batch length/div_sa number_filter, C
        x = x.reshape(x.shape[0], x.shape[1], -1).contiguous()
        x = self.dropout(x)
        x = self.activation(self.fc(x))
        return x

class Time_Embedding(nn.Module):
    def __init__(
        self,
        input_shape ,
        token_d_model , 
        filter_num = 16, filter_size=5, dw_layers = 3 ,se_layers = 2,
    ):
        super(Time_Embedding, self).__init__()
        print("Time_Embedding")
        c_in = input_shape[2]
        layers_dw = []
        for i in range(dw_layers):
            if i == 0:
                in_channel = 1
            else:
                in_channel = filter_num
    
            layers_dw.append(nn.Sequential(
                nn.Conv2d(in_channel, filter_num, (filter_size, 1), padding ="same"),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(filter_num),

            ))# B F L C
        self.layers_dw = nn.ModuleList(layers_dw)
        layers_se = []
        for _ in range(se_layers):
            layers_se.append( SE_Block(c_in, 1))
            layers_se.append(nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, padding ="same",groups=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(c_in)
            ))
            layers_se.append(nn.Sequential(
                nn.Conv2d(c_in, c_in, filter_size, padding ="same",groups=c_in),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(c_in),
                nn.MaxPool2d(3, stride=(2,1),padding=(1,1))
            ))
        # B C F L

        self.layers_se = nn.ModuleList(layers_se)
        
        shape = self.get_the_shape(input_shape)
        
        self.fc = nn.Linear(shape[2],1)
        self.activation = nn.ReLU()

        self.fc1 = nn.Linear(shape[1],36)
        self.fc2 = nn.Linear(36,token_d_model)


    def get_the_shape(self, input_shape):
        x = torch.rand(input_shape)
        x = x.unsqueeze(1)
        for layer in self.layers_dw:
            x = layer(x)      
        x = x.permute(0,3,1,2)
        for layer in self.layers_se:
            x = layer(x)  
        return x.shape
    
    def forward(self, x):
        # B L C 输入的各式
        x = x.unsqueeze(1)
        # B 1 L C
        for layer in self.layers_dw:
            x = layer(x)      
        # B F L C
        x = x.permute(0,3,1,2)
        # B C F L
        for layer in self.layers_se:
            x = layer(x) 
        # B C F L

        x = self.activation(torch.squeeze(self.fc(x.permute(0,3,1,2)),3)) # B L C


        x = self.activation(self.fc1(x))
        y = self.activation(self.fc2(x))     
        return y

class TimeEmbedder(nn.Module):
    def __init__(self, args ):
        super(TimeEmbedder, self).__init__()

        #self.value_embedding = TokenEmbedding(c_in                 = args.c_in, 
        #                                      token_d_model        = args.token_d_model,
        #                                      kernel_size          = args.token_kernel_size, 
        #                                      stride               = args.token_stride, 
        #                                      conv_bias            = args.token_conv_bias,
        #                                      activation           = args.token_activation,
        #                                      norm_type            = args.token_norm,
        #                                      n_conv_layers        = args.token_n_layers,
        #                                      in_planes            = args.token_in_planes,
        #                                      max_pool             = args.token_max_pool,
        #                                      pooling_kernel_size  = args.token_pool_kernel_size, 
        #                                      pooling_stride       = args.token_pool_stride,
        #                                      pooling_padding      = args.token_pool_pad,
        #                                      padding_mode         = args.padding_mode,
        #                                      light_weight         = args.light_weight)
        #self.value_embedding = Time_Embedding( input_shape = (1, args.input_length, args.c_in) ,
        #                                       token_d_model = args.token_d_model )
        self.value_embedding = Time_FeatureExtractor_attention( input_shape = (1, args.input_length, args.c_in) ,
                                                                number_filter = 16, token_d_model=args.token_d_model )
        #sequence_length = self.value_embedding.sequence_length(length       =  args.input_length,   n_channels   =  args.c_in) + 1
        sequence_length = int(np.ceil(args.input_length/div_sa)) + 1
        
        self.class_emb = nn.Parameter(torch.zeros(1, 1, args.token_d_model), requires_grad=True)

        if args.positional_embedding != 'none':
            
            if args.positional_embedding == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, sequence_length, args.token_d_model), requires_grad=True)
                # nn.Parameter(torch.randn(1, num_patches + 1, dim))
                nn.init.trunc_normal_(self.pos_embedding, std=0.2)
        
            else:
                self.pos_embedding = nn.Parameter(self.sinusoidal_embedding(sequence_length, args.token_d_model), requires_grad=False)
        else:
            self.pos_embedding = None
            
        self.input_embedding_dropout = nn.Dropout(p = args.input_embedding_dropout) 

    def forward(self, x):


        x = self.value_embedding(x)


        cls_token = self.class_emb.expand(x.shape[0], -1, -1)
        x = torch.cat((x,cls_token), dim=1)

        if self.pos_embedding is not None:
            x += self.pos_embedding

        x = self.input_embedding_dropout(x)
        
        return x


# ---------------------- Time Freq input Embdding ----------------------


class Freq_Forward_block(nn.Module):
    def __init__(self, 
                 c_in, 
                 c_out,  # 主要是把channel的dim压平
                 kernel_size, 
                 stride=1, 
                 bias = False, 
                 padding_mode = "replicate"):
        
        super(Freq_Forward_block, self).__init__()
        
        # depthwise
        self.dw_conv = nn.Conv2d(in_channels  = c_in,
                                 out_channels = c_in,
                                 kernel_size  = [kernel_size,kernel_size],
                                 padding      = [int(kernel_size/2),int(kernel_size/2)],
                                 groups       = c_in,
                                 stride       = [1,stride],  #缩短长度
                                 bias         = bias,  
                                 padding_mode = padding_mode)
        self.batch_norm_1 = nn.BatchNorm2d(c_in)
        self.act_1  = nn.ReLU()
        # pointwise
        self.pw_conv = nn.Conv2d(in_channels  = c_in,
                                 out_channels = c_out,    # 压平
                                 kernel_size  = 1,
                                 padding      = 0,
                                 stride       = 1,
                                 bias         = bias,  
                                 padding_mode = padding_mode)
        self.batch_norm_2 = nn.BatchNorm2d(c_out)
        self.act_2  = nn.ReLU()
        
    def forward(self, x):

        x  = self.dw_conv(x)
        x  = self.batch_norm_1(x)
        x  = self.act_1(x)

        x  = self.pw_conv(x)
        x  = self.batch_norm_2(x)
        x  = self.act_2(x)

        return x




class Freq_TokenEmbedding(nn.Module):
    def __init__(self,
                 c_in, 
                 token_d_model,
                 kernel_size            = 3, 
                 stride                 = 1,  #横向方向缩短距离
                 conv_bias              = False,
                 n_conv_layers          = 1,
                 sampling_freq          = 100,
                 padding_mode           = 'replicate',
                 light_weight           = False):
        """
        c_in  : 模型输入的维度
        token_d_model ： embedding的维度  TODO看看后面是需要被相加还是被cat
        kernel_size   : 每一层conv的kernel大小
    
        """
        super(Freq_TokenEmbedding, self).__init__()

        n_filter_list = [c_in] + [max(1,int(c_in/2**(i+1))) for i in range(n_conv_layers - 1)] + [1]
        #print(n_filter_list)
        self.conv_layers = []
        for i in range(n_conv_layers):
            self.conv_layers.append(Freq_Forward_block(c_in           = n_filter_list[i], 
                                                       c_out          = n_filter_list[i + 1],  # 主要是把channel的dim压平
                                                       kernel_size    = kernel_size, 
                                                       stride         = stride, 
                                                       bias           = conv_bias,
                                                       padding_mode   = padding_mode))

        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.conv = nn.Conv1d(in_channels  =  self.channel(c_in = c_in, freq = sampling_freq, length=100), 
                              out_channels =  token_d_model,
                              kernel_size  =  kernel_size,
                              padding      =  int(kernel_size/2),
                              stride       =  1,
                              bias         =  conv_bias,
                              padding_mode =  padding_mode)
        self.norm        = nn.LayerNorm(token_d_model)
        self.activation  = nn.ReLU()
    def forward(self, x):

        #x = x.permute(0, 2, 1, 3)
        for layer in self.conv_layers:
            x = layer(x)

        x = torch.squeeze(x, 1)

        x = self.conv(x) # B C L
        x = self.activation(self.norm(x.permute(0, 2, 1)))

        return x
    
    def sequence_length(self, c_in = 100, freq = 50, length=100):
        x =  torch.rand(1,c_in,freq,length).float()
        for layer in self.conv_layers:
            x = layer(x)
        return x.shape[3]

    def channel(self, c_in = 100, freq = 50, length=100):
        x =  torch.rand(1,c_in,freq,length).float()
        for layer in self.conv_layers:
            x = layer(x)
        #print("channel ,", x.shape[2])
        return x.shape[2]

class TimeFreq_FeatureExtractor_attention(nn.Module):
    def __init__(self, input_shape, number_filter, token_d_model, conv_number=4, filter_size=5, activation = "ReLU"):
        super(TimeFreq_FeatureExtractor_attention, self).__init__()
        # input_shape: B C F L
        print("TimeFreq_FeatureExtractor_attention with ",div_sa)
        c_in = input_shape[1]
        f_in = input_shape[2]
        layers = []
        for i in range(conv_number):
            if i == 0:
                temp = f_in
            else:
                temp = number_filter
            layers.append(nn.Sequential(
                nn.Conv2d(temp, number_filter, (filter_size, 1), padding ="same"),
                nn.ReLU(inplace=True),
                #nn.BatchNorm2d(number_filter)
            ))
        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(0.2)

        self.sa = SelfAttention(number_filter*div_sa, div=div_sa)
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()
        self.fc = nn.Linear(number_filter*c_in, token_d_model)

    def forward(self, x):
        # x shape batch Channel Freq Length
        x = x.permute(0,2,3,1)
        # x shape batch Freq Length Channel
        for layer in self.layers:
            x = layer(x)
        # x shape batch number_filter, Length Channel
        batch, filter, length, channel = x.shape
        #print("freq x shape before attention : " , x.shape)

        #refined = torch.cat(
        #    [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
        #    dim=-1,
        #)
        #x = refined.permute(0, 3, 1, 2)
        #x = x.reshape(x.shape[0], x.shape[1], -1)
        #x = self.dropout(x)
        #x = self.activation(self.fc(x))
        refined = [] 
        step = int(np.ceil(length/div_sa))

        for index in range(step):
            if index<step-1:
                temp = torch.unsqueeze(x[:, :, index*div_sa:(index+1)*div_sa, :].reshape(batch, -1, channel).contiguous(), dim=3)
            else:
                temp = torch.unsqueeze(x[:, :, -div_sa:, :].reshape(batch, -1, channel).contiguous(), dim=3)
            refined.append(self.sa(temp))

        refined = torch.cat(refined,   dim=-1)
        # refined shape batch numberfilter, C, Length/div_sa
        x = self.activation(refined.permute(0, 3, 1, 2))
        #print("freq x shape after attention : " , x.shape)
        # x shape batch length/3 number_filter, C
        x = x.reshape(x.shape[0], x.shape[1], -1).contiguous()
        x = self.dropout(x)
        x = self.activation(self.fc(x))

        return x



class TimeFreq_TokenEmbedding(nn.Module):
    def __init__(self, input_shape, token_d_model, dw_layers = 3 ,se_layers = 3):
        super(TimeFreq_TokenEmbedding, self).__init__()
        c_in = input_shape[1]
        layers = []
        for _ in range(dw_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(c_in, c_in, 5, padding ="same",groups=c_in),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(c_in),
                #nn.MaxPool2d(3, stride=(2,1),padding=(1,1))
            ))
        

        for _ in range(se_layers):
            layers.append( SE_Block(c_in, 1))
            layers.append(nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, padding ="same",groups=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(c_in)
            ))
            layers.append(nn.Sequential(
                nn.Conv2d(c_in, c_in, 5, padding ="same",groups=c_in),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(c_in),
                nn.MaxPool2d(3, stride=(2,1),padding=(1,1))
            ))
        

        self.layers = nn.ModuleList(layers)
        shape = self.get_the_shape(input_shape) # B C F L
		
        dim = shape[1]*shape[2]
        #print("------------dim-------------", shape[1], "    ", shape[2])
        self.fc = nn.Linear(dim,int(dim/2))
        self.activation = nn.ReLU()

        #self.fc1 = nn.Linear(shape[1],36)
        self.fc2 = nn.Linear(int(dim/2),token_d_model)

    def get_the_shape(self, input_shape):
        x = torch.rand(input_shape)
        for layer in self.layers:
            x = layer(x)        
        return x.shape
    
    def forward(self, x):
        # 输入的各式为 B C Freq L
        #x = x.permute(0,3,1,2)
        b,_,_,L = x.shape
        for layer in self.layers:
            x = layer(x)   

        #x = self.activation(torch.squeeze(self.fc(x.permute(0,3,1,2)),3)) # B L C
        x = x.permute(0,3,1,2).view(b,L,-1)
        x = x.contiguous()
        x = self.activation(self.fc(x))
        y = self.activation(self.fc2(x))     
        return y

class FreqEmbedder(nn.Module):
    def __init__(self, args ):
        super(FreqEmbedder, self).__init__()
        
        #self.value_embedding = Freq_TokenEmbedding(c_in            = args.c_in,  
        #                                           token_d_model   = args.token_d_model,
        #                                           kernel_size     = args.token_kernel_size, 
        #                                           stride          = args.token_stride,
        #                                           conv_bias       = args.token_conv_bias,
        #                                           n_conv_layers   = args.token_n_layers,
        #                                           sampling_freq   = args.sampling_freq,
        #                                           padding_mode    = args.padding_mode,
        #                                           light_weight           = False)
        if args.windowsize >= 60:
            l_scale = 2
        else:
            l_scale = 1
        if args.sampling_freq >=40:
            f_scale = 2
        else:
            f_scale = 1

        #self.value_embedding = TimeFreq_TokenEmbedding( input_shape   = (1, args.c_in, int(args.sampling_freq/f_scale), int(args.input_length/l_scale)),
        #                                                token_d_model = args.token_d_model,
        #                                                dw_layers = 3 ,se_layers = 2)


        self.value_embedding = TimeFreq_FeatureExtractor_attention(input_shape   = (1, args.c_in, int(args.sampling_freq/f_scale), int(args.input_length/l_scale)),
                                                                   number_filter = 32,
                                                                   token_d_model = args.token_d_model,
                                                                   conv_number   = 4, filter_size=5)

        #sequence_length = self.value_embedding.sequence_length(c_in = args.c_in, freq = args.sampling_freq, length= args.input_length)+1
        #sequence_length = int(args.input_length/l_scale) + 1
        sequence_length = int(np.ceil(int(args.input_length/l_scale)/div_sa)) + 1

        self.class_emb = nn.Parameter(torch.zeros(1, 1, args.token_d_model), requires_grad=True)

        if args.positional_embedding != 'none':
            
            if args.positional_embedding == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, sequence_length, args.token_d_model), requires_grad=True)
                # nn.Parameter(torch.randn(1, num_patches + 1, dim))
                nn.init.trunc_normal_(self.pos_embedding, std=0.2)
        
            else:
                self.pos_embedding = nn.Parameter(self.sinusoidal_embedding(sequence_length, args.token_d_model), requires_grad=False)
        else:
            self.pos_embedding = None
            
        self.input_embedding_dropout = nn.Dropout(p = args.input_embedding_dropout) 

    def forward(self, x):

        #print(x.shape)
        x = self.value_embedding(x)

        cls_token = self.class_emb.expand(x.shape[0], -1, -1)
        x = torch.cat((x,cls_token), dim=1)


        if self.pos_embedding is not None:
            x += self.pos_embedding

        x = self.input_embedding_dropout(x)
        
        return x










