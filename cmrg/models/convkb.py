import torch.nn as nn
from cmrg.config.cuda import *

set_random_seed(1234)


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len)).cuda()  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear(input_dim * out_channels, 1).cuda()
        self.bn1 = torch.nn.BatchNorm1d(50)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)
        x = self.dropout(conv_input)
        x = self.conv_layer(x)
        x = self.bn1(x.squeeze(-1))
        out_conv = torch.relu(x)
        # out_conv = self.dropout(
        #     self.non_linearity(self.conv_layer(conv_input)))
        input_fc = out_conv.view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output
