import torch.nn as nn
from cmrg.models.gcn_encoder import GCN_Encoder
from cmrg.models.attention import Attention_out
from cmrg.models.spgat import SpGAT
from cmrg.models.dgi import DGI
from cmrg.utils.func import batch_gat_loss, load_data, save_model, norm_embeddings, read_edge_index, batch_graph_gen, \
    gen_shuf_fts, gen_txt_file, load_graph
from cmrg.models.convkb import ConvKB
from cmrg.config.cuda import *
from torch_scatter import scatter_max, scatter_sum, scatter_mean, scatter_softmax, scatter_add, scatter_min
class CMRG(nn.Module):
    def __init__(self, args, nfeat, nhid1, nhid2, dropout, initial_entity_emb,
                 initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT, ft_size, hid_units, nonlinearity, Corpus_):
        super(CMRG, self).__init__()
        self.SGCN1 = GCN_Encoder(400, nhid1, 200, dropout)
        self.SGCN2 = GCN_Encoder(200, nhid1, 200, dropout)
        self.dropout = dropout
        self.attention_out = Attention_out(200, args.attention_hidden_size)
        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]
        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]
        self.drop_GAT = drop_GAT
        self.alpha = alpha  # For leaky relu
        self.final_entity_embeddings = nn.Parameter(torch.randn(initial_entity_emb.shape[0], 200))
        self.final_relation_embeddings = nn.Parameter(torch.randn(initial_relation_emb.shape[0], 200))
        self.DGI = DGI(200, hid_units, nonlinearity)
        self.relation_embeddings_ = nn.Parameter(torch.zeros(
            size=(initial_relation_emb.shape[0], 200)))
        nn.init.xavier_uniform_(self.relation_embeddings_.data, gain=1.414)
        self.entity_embeddings_ = nn.Parameter(torch.zeros(
            size=(initial_entity_emb.shape[0], 200)))
        nn.init.xavier_uniform_(self.entity_embeddings_.data, gain=1.414)
        self.layer_emb = nn.Sequential(
            nn.Linear(400, 1),
        )
        self.layer_emb_out = nn.Sequential(
            nn.Linear(400, 200),
            nn.Dropout(self.dropout)
        )
        self.b_x, self.b_node_graph_index, self.b_edge_index, self.b_new_adj = batch_graph_gen(
            Corpus_.new_entity2id, args)
        self.convKB = ConvKB(200, 3, 1, 50, 0, 0.2)
        self.m = nn.Softmax(dim=1)
        self.bn = torch.nn.BatchNorm1d(200)
        self.big_adj = load_graph(args)

    def multi_context_encoder(self, entity_feat, rel_feat):
        new_entity_rel_embed = torch.cat(
            [entity_feat[self.b_x], rel_feat[self.b_node_graph_index]], dim=-1)
        entity_embed = self.SGCN1(new_entity_rel_embed.cuda(), self.b_new_adj.cuda())
        index = torch.tensor(self.b_x).long().cuda()
        out = scatter_mean(entity_embed, index, dim=0)
        z = out[index]
        emb = torch.cat([entity_embed[index], z], dim=-1)
        new_emb = self.layer_emb(emb)
        z_s = scatter_softmax(new_emb, index, dim=0)
        new_out = scatter_add(z_s * emb, index, dim=0)
        new_out = self.layer_emb_out(new_out) + entity_feat
        return new_out, rel_feat
    def high_order_encoder(self, entity_feat):
        entity_embed = self.SGCN2(entity_feat, self.big_adj.cuda())
        return entity_embed
    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop, sparse):
        new_entity_embed = self.entity_embeddings_
        new_entity_embed = norm_embeddings(new_entity_embed)
        new_rel_embed = self.relation_embeddings_
        new_rel_embed = norm_embeddings(new_rel_embed)
        new_entity_embed_ = gen_shuf_fts(new_entity_embed)
        entity_con, rel_con = self.multi_context_encoder(new_entity_embed, new_rel_embed)
        entity_con_, rel_con_ = self.multi_context_encoder(new_entity_embed_, new_rel_embed)
        entity_global = self.high_order_encoder(entity_con)
        entity_global_ = self.high_order_encoder(entity_con_)
        new_entity_embed = entity_con
        new_rel_embed = rel_con
        local_logits = self.DGI(entity_con[np.newaxis],  entity_con_[np.newaxis], sparse, None, None, None)
        global_logits = self.DGI(entity_global[np.newaxis], entity_global_[np.newaxis], sparse, None, None, None)
        self.final_entity_embeddings.data = new_entity_embed.data
        self.final_relation_embeddings.data = new_rel_embed.data
        conv_input = torch.cat((new_entity_embed[batch_inputs[:, 0], :].unsqueeze(1),
                                new_rel_embed[batch_inputs[:, 1]].unsqueeze(1),
                                new_entity_embed[batch_inputs[:, 2], :].unsqueeze(1)),
                               dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv, local_logits, global_logits

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1),
                                self.final_relation_embeddings[batch_inputs[:, 1]].unsqueeze(1),
                                self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)),
                               dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv