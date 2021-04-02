import torch.nn as nn
from cmrg.config.cuda import *
set_random_seed(1234)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, seq, msk):
        if msk is None:
            _ = torch.mean(seq, 1)
            return _
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.dropout = 0.1
        self.disc = Discriminator(n_in)

    def forward(self, seq1_top, seq2_top, sparse, msk, samp_bias1, samp_bias2):
        h_1_top_node = seq1_top
        c_top_graph = self.read(h_1_top_node, msk)
        c_top_graph = self.sigm(c_top_graph)
        h_2_top_node = seq2_top
        ret = self.disc(c_top_graph, h_1_top_node, h_2_top_node, samp_bias1, samp_bias2)
        return ret
