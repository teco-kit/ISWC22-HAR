import torch
import torch.nn as nn
import torch.nn.functional as F

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

        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels//div)
            self.key = conv1d(n_channels, n_channels//div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class TemporalAttention(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc(x).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 0)
        return context


def init_weights_orthogonal(m):
    """
    Orthogonal initialization of layer parameters
    :param m:
    :return:
    """
    if type(m) == nn.LSTM or type(m) == nn.GRU:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    elif type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, z):
        return self.fc(z)

class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        activation,
        sa_div,
    ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[1], filter_num, (filter_size, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))

        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(
            filter_num * input_shape[3],
            hidden_dim,
            enc_num_layers,
            bidirectional=enc_is_bidirectional,
            dropout=dropout_rnn,
        )

        self.ta = TemporalAttention(hidden_dim)
        self.sa = SelfAttention(filter_num, sa_div)

    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        x = refined.permute(3, 0, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        outputs, h = self.rnn(x)

        # apply temporal attention on GRU outputs
        out = self.ta(outputs)
        return out
    

class AttendDiscriminate(nn.Module):
    def __init__(
        self,
        input_shape,
        num_class,
        filter_scaling_factor=1,
        config = None):
        #hidden_dim = 128,
        #filter_num = 64,
        #filter_size = 5,
        #enc_num_layers = 2,
        #enc_is_bidirectional = False,
        #dropout = 0.5,
        #dropout_rnn = 0.25,
        #dropout_cls = 0.5,  #OPPO 
        #activation = "ReLU",
        #sa_div = 1,
        super(AttendDiscriminate, self).__init__()

        self.hidden_dim = int(filter_scaling_factor*config["hidden_dim"])
        self.filter_num = int(filter_scaling_factor*config["filter_num"])

        self.filter_size          = config["filter_size"]
        self.enc_num_layers       = config["enc_num_layers"]
        self.enc_is_bidirectional = False
        self.dropout              = config["dropout"]
        self.dropout_rnn          = config["dropout_rnn"]
        self.dropout_cls          = config["dropout_cls"]
        self.activation           = config["activation"]
        self.sa_div               = config["sa_div"]


        self.fe = FeatureExtractor(
            input_shape,
            self.hidden_dim,
            self.filter_num,
            self.filter_size,
            self.enc_num_layers,
            self.enc_is_bidirectional,
            self.dropout,
            self.dropout_rnn,
            self.activation,
            self.sa_div,
        )

        self.dropout = nn.Dropout(self.dropout_cls)
        self.classifier = Classifier(self.hidden_dim, num_class)



    def forward(self, x):
        feature = self.fe(x)
        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        out = self.dropout(feature)
        logits = self.classifier(out)
        return  logits
