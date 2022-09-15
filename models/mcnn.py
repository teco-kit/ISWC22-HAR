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




class MCNN(nn.Module):
    def __init__(self, 
                 input_shape, 
                 nb_classes,
                 filter_scaling_factor,
                 config):
                 #nb_conv_blocks        = 2,
                 #nb_filters            = 64,
                 #dilation              = 1,
                 #batch_norm            = False,
                 #filter_width          = 5,
                 #nb_layers_lstm        = 2,
                 #drop_prob             = 0.5,
                 #nb_units_lstm         = 128):
        """
        DeepConvLSTM model based on architecture suggested by Ordonez and Roggen (https://www.mdpi.com/1424-8220/16/1/115)
        
        """
        super(MCNN, self).__init__()
        self.nb_conv_blocks = config["nb_conv_blocks"]
        self.nb_filters     = int(config["nb_filters"]*filter_scaling_factor)
        self.dilation       = config["dilation"]
        self.batch_norm     = bool(config["batch_norm"])
        self.filter_width   = config["filter_width"]
        self.drop_prob      = config["drop_prob"]

        
        
        self.nb_channels = input_shape[3]
        self.nb_classes = nb_classes

    
        self.conv_blocks = []

        for i in range(self.nb_conv_blocks):
            if i == 0:
                input_filters = input_shape[1]
            else:
                input_filters = self.nb_filters
    
            self.conv_blocks.append(ConvBlock(self.filter_width, input_filters, self.nb_filters, self.dilation, self.batch_norm))

        
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        shape = self.get_the_shape(input_shape)
        final_length  = shape[2]
        # B F L* C

        # define dropout layer
        self.dropout = nn.Dropout(self.drop_prob)
        
        # Sensor Fusion
        self.activation = nn.ReLU() 
        self.fc_sensor_fusion = nn.Linear(self.nb_filters*self.nb_channels ,2*self.nb_filters)
        
        # Temporal Fusion   
        self.flatten = nn.Flatten()
        self.fc_temporal_fusion = nn.Linear(2*self.nb_filters*final_length ,self.nb_filters*2)
        
        # define classifier
        self.fc_prediction = nn.Linear(self.nb_filters*2, self.nb_classes)


    def get_the_shape(self, input_shape):
        x = torch.rand(input_shape)
        for conv_block in self.conv_blocks:
            x = conv_block(x)    

        return x.shape

    def forward(self, x):


        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
        # B F L* C

    
        x = x.permute(0, 2, 1, 3)
        # B L*  F C

        x = x.reshape(x.shape[0], x.shape[1], self.nb_filters * self.nb_channels)
        x = self.dropout(x)
        # B L*  F*C

        x = self.activation(self.fc_sensor_fusion(x)) 
        # B L*  2*C

        x = self.flatten(x)
        x = self.activation(self.fc_temporal_fusion(x)) # B L C


        out = self.fc_prediction(x)    

        return out

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)