import torch
import os
import pickle
import numpy as np
import tf_geometric as tfg
import scipy.sparse as sp
from tqdm import tqdm
from cmrg.data_process.create_batch import Corpus
from cmrg.data_process.preprocess import init_embeddings, build_data

from cmrg.config.cuda import *
set_random_seed(1234)

def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed, args):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=-1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=-1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def save_model(model, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained_{}.pth").format(epoch))
    print("Done saving Model")


def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
        args.data, is_unweigted=False, directed=True)

    if not args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, "img_text.pickle"),
                                                                 os.path.join(args.data, "rel_text.pickle"))

        print("Initialised relations and entities from Pre-train")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")

    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop,
                    )

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)


def norm_embeddings(embeddings: torch.Tensor):
    norm = embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings / norm


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def normalize2(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize3(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def gen_adj(sample_struct_edges):
    shape_size = max([max(sample_struct_edges[:, 0]), max(sample_struct_edges[:, 1])]) + 1
    sedges = np.array(list(sample_struct_edges), dtype=np.int32).reshape(sample_struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(shape_size, shape_size),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return nsadj


def batch_graph_gen(new_entity2id, args):
    edge_index = read_edge_index(args)
    big_graph = []
    entity_index = []
    for key, val in new_entity2id.items():
        temp = []
        for k in val.keys():
            temp.append(k)
        entity_index.append(temp)
    for i in range(len(new_entity2id)):
        sub_graph = tfg.Graph(x=entity_index[i], edge_index=edge_index[i])
        big_graph.append(sub_graph)
    batch_graph = tfg.BatchGraph.from_graphs(big_graph)
    new_edges = []
    edges_data = batch_graph.edge_index
    for i in range(len(edges_data[0])):
        new_edges.append([edges_data[0][i], edges_data[1][i]])
    new_adj = gen_adj((np.array(new_edges)))
    return (batch_graph.x), (batch_graph.node_graph_index), (batch_graph.edge_index), (
        new_adj)
def specify_node_update(rel_dic):
        rel_entity_set = []
        for tris in rel_dic.values():
            temp = []
            for tri in tris:
                if tri[0] not in temp:
                    temp.append(tri[0])
                if tri[1] not in temp:
                    temp.append(tri[1])
            rel_entity_set.append(temp)
        new_entity2id = {}
        for i, item in enumerate(rel_entity_set):
            new_entity2id[i] = {}
            for j, _ in enumerate(item):
                new_entity2id[i][_] = j
        return new_entity2id

def gen_intra_context_edges(batch_inputs,args):
    rel_dic = {}
    for i in range(279):
        temp = []
        for tri in batch_inputs:
            if tri[1] == i and [tri[0], tri[-1]] not in temp:
                temp.append([tri[0], tri[-1]])
        rel_dic[i] = np.array(temp)
    return rel_dic
def batch_graph_gen2(new_entity2id, batch_inputs, args):
    rel_dic = gen_intra_context_edges(batch_inputs, args)
    new_entity2id = specify_node_update(rel_dic)

    edge_index = []
    for key, value in new_entity2id.items():
        temp = value
        edge_index.append(temp)
    big_graph = []
    entity_index = []
    for key, val in new_entity2id.items():
        temp = []
        for k in val.keys():
            temp.append(k)
        entity_index.append(temp)
    for i in range(len(new_entity2id)):
        sub_graph = tfg.Graph(x=entity_index[i], edge_index=edge_index[i])
        big_graph.append(sub_graph)
    batch_graph = tfg.BatchGraph.from_graphs(big_graph)
    new_edges = []
    edges_data = batch_graph.edge_index
    for i in range(len(edges_data[0])):
        new_edges.append([edges_data[0][i], edges_data[1][i]])
    new_adj = gen_adj((np.array(new_edges)))
    return (batch_graph.x), (batch_graph.node_graph_index), (batch_graph.edge_index), (
        new_adj)


def read_edge_index(args):
    file = args.data + "edge_index3.pickle"
    with open(file, 'rb') as handle:
        edge_index = pickle.load(handle)
    return edge_index


def big_graph_edges_gen(batch_graph):
    ori = batch_graph.edge_index.numpy()[0]
    dst = batch_graph.edge_index.numpy()[1]
    l = len(ori)
    with open("big_graph_edges.txt", 'a', encoding="utf-8") as f:
        for i in range(l):
            temp = str(ori[i]) + "\t" + str(dst[i]) + "\n"
            f.write(temp)


def edge_index_gen(new_entity2id, new_1hop):
    edge_index = []
    count = 0
    for key, val in tqdm(new_entity2id.items()):
        print(count)
        count += 1
        ori = []
        dst = []
        entity_new_id = []
        entity_ori_id = []
        for tri_key, tri_val in val.items():
            entity_new_id.append(tri_val)
            entity_ori_id.append(tri_key)
        if len(entity_new_id) == 2:
            edge_index.append(np.array([[entity_new_id[0]], [entity_new_id[1]]]))
        else:
            temp_tris = []
            for key_, val_ in new_1hop.items():
                if key_ in entity_ori_id:
                    for tri in val_:
                        if tri[1] == key and tri not in temp_tris:
                            temp_tris.append(tri)
            for tri in temp_tris:
                ori.append(val[tri[0]])
                dst.append(val[tri[2]])
            edge_index.append(np.array([ori, dst]))
    file = "edge_index3.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(edge_index, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
def load_graph(args):
    featuregraph_path = args.data + 'edges_train.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])),
                         shape=(args.entity_num, args.entity_num),
                         dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return nfadj


def gen_shuf_fts(entity):
    idx = np.random.permutation(len(entity))
    shuf_fts = entity[idx, :] + torch.randn(entity.shape).cuda()
    return shuf_fts

def gen_txt_file(filepath, filename):
    if os.path.exists(filepath):
        print("The dir is already exists")
    else:
        os.makedirs(filepath)
    if os.path.exists(filename):
        print("The file is already exists")
    else:
        res_file = open(filename, "w")
        res_file.close()
