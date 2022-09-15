
import torch.nn as nn


# Lightweight conv
class DW_PW_projection(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, bias = False, padding_mode = "replicate"):
        super(DW_PW_projection, self).__init__()

        self.dw_conv1d = nn.Conv1d(in_channels  = c_in,
                                   out_channels = c_in,
                                   kernel_size  = kernel_size,
                                   padding      = int(kernel_size/2),
                                   groups       = c_in,
                                   bias         = bias,  
                                   padding_mode = padding_mode)

        self.pw_conv1d = nn.Conv1d(in_channels  = c_in,
                                   out_channels = c_out,
                                   kernel_size  = 1,
                                   padding      = 0,
                                   groups       = 1,
                                   bias         = bias,  
                                   padding_mode = padding_mode)
    def forward(self, x):


        x  = self.dw_conv1d(x)
        x  = self.pw_conv1d(x)

        return x


Norm_dict = {"layer" : nn.LayerNorm,
             "batch" : nn.BatchNorm1d}


Activation_dict = {"relu"         : nn.ReLU,
                   "leakyrelu"    : nn.LeakyReLU,
                   "prelu"        : nn.PReLU,
                   "rrelu"        : nn.RReLU,
                   "elu"          : nn.ELU,
                   "gelu"         : nn.GELU,
                   "hardswish"    : nn.Hardswish,
                   "mish"         : nn.Mish}