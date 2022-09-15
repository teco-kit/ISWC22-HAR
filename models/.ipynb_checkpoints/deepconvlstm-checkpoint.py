import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.conv1 = nn.Conv2d(self.input_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1), dilation=(self.dilation, 1), stride=(2,1))
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

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


class DeepConvLSTM(nn.Module):
    def __init__(self, 
                 input_shape, 
                 nb_classes,
                 filter_scaling_factor,
                 config):
                 #nb_conv_blocks         = 2,
                 #nb_filters             = 64,
                 #dilation               = 1,
                 #batch_norm             = False,
                 #filter_width           = 5,
                 #nb_layers_lstm         = 1,
                 #drop_prob              = 0.5,
                 #nb_units_lstm          = 128):
        """
        DeepConvLSTM model based on architecture suggested by Ordonez and Roggen (https://www.mdpi.com/1424-8220/16/1/115)
        
        """
        super(DeepConvLSTM, self).__init__()
        self.nb_conv_blocks = config["nb_conv_blocks"]
        self.nb_filters     = int(filter_scaling_factor*config["nb_filters"])
        self.dilation       = config["dilation"]
        self.batch_norm     = bool(config["batch_norm"])
        self.filter_width   = config["filter_width"]
        self.nb_layers_lstm = config["nb_layers_lstm"]
        self.drop_prob      = config["drop_prob"]
        self.nb_units_lstm  = int(filter_scaling_factor*config["nb_units_lstm"])
        
        
        self.nb_channels    = input_shape[3]
        self.nb_classes     = nb_classes

    
        self.conv_blocks = []

        for i in range(self.nb_conv_blocks):
            if i == 0:
                input_filters = input_shape[1]
            else:
                input_filters = self.nb_filters
    
            self.conv_blocks.append(ConvBlock(self.filter_width, input_filters, self.nb_filters, self.dilation, self.batch_norm))

        
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        
        # define lstm layers
        self.lstm_layers = []
        for i in range(self.nb_layers_lstm):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(self.nb_channels * self.nb_filters, self.nb_units_lstm, batch_first =True))
            else:
                self.lstm_layers.append(nn.LSTM(self.nb_units_lstm, self.nb_units_lstm, batch_first =True))
        self.lstm_layers = nn.ModuleList(self.lstm_layers)
        
        # define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        # define classifier
        self.fc = nn.Linear(self.nb_units_lstm, self.nb_classes)

    def forward(self, x):
        # reshape data for convolutions
        # B,L,C = x.shape
        # x = x.view(B, 1, L, C)

        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
        final_seq_len = x.shape[2]

        # permute dimensions and reshape for LSTM
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, final_seq_len, self.nb_filters * self.nb_channels)

        x = self.dropout(x)
        

        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
            

        x = x[:, -1, :]
    
        x = self.fc(x)


        return x

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)