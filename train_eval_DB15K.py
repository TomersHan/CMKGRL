# coding = utf-8
import torch
import argparse
import sys
import time
import random
import pickle
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from cmrg.utils.func import batch_gat_loss, load_data, save_model, norm_embeddings, read_edge_index, batch_graph_gen, \
    gen_shuf_fts, gen_txt_file, load_graph
from cmrg.config.cuda import *
from cmrg.models.cmrg import CMRG
def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/FB15k-237/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3600, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=True, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=100, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=True)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=True)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/db/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=86835, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layers")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layers")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    # args.add_argument("-margin", "--margin", type=float,
    #                   default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layers")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=50,
                      help="Number of output channels in conv layers")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layers")

    # Single Attention
    args.add_argument("-attention_hidden_size", "--attention_hidden_size", type=int,
                      default=16, help="Attention_hiden_size for attention layers")
    args.add_argument("-entity_num", "--entity_num", type=int,
                      default=12842, help="entity_num")
    args.add_argument("-model_name", "--model_name", type=str,
                      default="Con_Test")
    args.add_argument("-cuda", "--cuda", type=int,
                      default=0, help="cuda_number")
    args.add_argument("-cl_rate", "--cl_rate", type=float,
                      default=0.1, help="Dropout probability for convolution layers")

    cmd = "--data ./data/DB15K/  " \
          "--model_name CMRG_DB15K " \
          "--cuda 5 " \
          "--cl_rate 0.001 " \
          "--epochs_gat 7001 " \
          "--valid_invalid_ratio_gat 2 " \
          "--weight_decay_gat 0.00001 " \
          "--batch_size_gat 10000 " \
          "--output_folder ./checkpoints/db/cmrg/ " \
          "--out_channels 50 " \
          "--lr 0.002 " \
          "--attention_hidden_size 128 "
    sys.argv += cmd.split()
    args = args.parse_args()
    return args




args = parse_args()
set_random_seed(1234)

CUDA = choonsr_cuda(args.cuda)

Corpus_, entity_embeddings, relation_embeddings = load_data(args)




def train_encoder(args):
    model_encoder = CMRG(args, nfeat=200, nhid1=400, nhid2=200, dropout=0.2,
                          initial_entity_emb=entity_embeddings, initial_relation_emb=relation_embeddings,
                          entity_out_dim=args.entity_out_dim, relation_out_dim=args.entity_out_dim,
                          drop_GAT=args.drop_GAT, alpha=args.alpha, nheads_GAT=args.nheads_GAT, ft_size=200,
                          hid_units=512, nonlinearity='prelu', Corpus_=Corpus_,
                          )

    if CUDA:  model_encoder.cuda()
    optimizer = torch.optim.Adam(model_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    margin_loss = torch.nn.SoftMarginLoss()
    b_xent = nn.BCEWithLogitsLoss()
    current_batch_2hop_indices = torch.tensor([])
    epoch_losses = []  # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))
    lbl = torch.cat((torch.ones(1, len(entity_embeddings)), torch.zeros(1, len(entity_embeddings))), 1)
    file_path = "./train_log/{}".format(args.model_name, args.model_name)
    file_name = "./train_log/{}/{}_log.txt".format(args.model_name, args.model_name)
    gen_txt_file(file_path, file_name)
    with open("./train_log/{}/{}_log.txt".format(args.model_name, args.model_name), "a",
              encoding="utf-8")as log_encoder:
        for epoch in tqdm(range(args.epochs_gat)):
            print("\nepoch-> ", epoch)
            random.shuffle(Corpus_.train_triples)
            Corpus_.train_indices = np.array(
                list(Corpus_.train_triples)).astype(np.int32)
            model_encoder.train()  # getting in training mode
            start_time = time.time()
            epoch_loss = []
            if len(Corpus_.train_indices) % args.batch_size_gat == 0:
                num_iters_per_epoch = len(Corpus_.train_indices) // args.batch_size_gat
            else:
                num_iters_per_epoch = (len(Corpus_.train_indices) // args.batch_size_gat) + 1
            for iters in (range(num_iters_per_epoch)):
                start_time_iter = time.time()
                train_indices, train_values = Corpus_.get_iteration_batch(iters)
                if CUDA:
                    train_indices = Variable(torch.LongTensor(train_indices)).cuda()
                    train_values = Variable(torch.FloatTensor(train_values)).cuda()
                    lbl = lbl.cuda()
                else:
                    train_indices = Variable(torch.LongTensor(train_indices))
                    train_values = Variable(torch.FloatTensor(train_values))
                # forward pass
                preds, local_logits, global_logits = model_encoder(Corpus_,
                                              Corpus_.train_adj_matrix_topolopy,
                                              train_indices,
                                              current_batch_2hop_indices, sparse=True)
                optimizer.zero_grad()
                tri_loss = margin_loss(preds.view(-1), train_values.view(-1))
                # CL_loss
                cl_loss_local = b_xent(local_logits, lbl)
                cl_loss_global = b_xent(global_logits, lbl)
                loss = tri_loss + args.cl_rate*cl_loss_local+args.cl_rate*cl_loss_global
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.data.item())
            scheduler.step()
            print("Epoch {} , average loss {} , epoch_time {}".format(
                epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
            epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
            temp = "Epoch: {}\ntri_loss: {} average loss: {}\nepoch_time: {} current_time:{} \n".format(epoch,
                                                                                                        tri_loss,
                                                                                                        sum(
                                                                                                            epoch_loss) / len(
                                                                                                            epoch_loss),
                                                                                                        time.time() - start_time,
                                                                                    current_time)
            log_encoder.write(temp)
            if epoch % 100 == 0 and epoch >= 4000:
                save_model(model_encoder, epoch, args.output_folder)
            if epoch % 100 == 0 and epoch >= 4000:
                model_encoder.eval()
                with torch.no_grad():
                    hits_10000, hits_1000, hits_500, \
                    hits_100, hits_10, hits_3, \
                    hits_1, mean_rank, mean_recip_rank = Corpus_.get_validation_pred(args, model_encoder,
                                                                                     Corpus_.unique_entities_train)

                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))

                    temp = "Current_time: {}\n" \
                           "hits_10000:  {}\n" \
                           "hits_1000:  {}\n" \
                           "hits_500:  {}\n" \
                           "hits_100:  {}\n" \
                           "hits_10:  {}\n" \
                           "hits_3:  {}\n" \
                           "hits_1:  {}\n" \
                           "mean_rank:  {}\n" \
                           "mean_recip_rank:  {}\n".format(str(current_time), hits_10000, hits_1000,
                                                           hits_500,
                                                           hits_100, hits_10, hits_3,
                                                           hits_1, mean_rank, mean_recip_rank)
                    log_encoder.write(temp)


def evaluate_test(args, unique_entities):
    model_encoder = CMRG(args, nfeat=200, nhid1=400, nhid2=200, dropout=0.2,
                          initial_entity_emb=entity_embeddings, initial_relation_emb=relation_embeddings,
                          entity_out_dim=args.entity_out_dim, relation_out_dim=args.entity_out_dim,
                          drop_GAT=args.drop_GAT, alpha=args.alpha, nheads_GAT=args.nheads_GAT, ft_size=200,
                          hid_units=512, nonlinearity='prelu', Corpus_=Corpus_,
                          )
    model_encoder.load_state_dict(torch.load(
        '{0}/trained_best_model.pth'.format(args.output_folder)), strict=False)

    model_encoder.cuda()
    model_encoder.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(args, model_encoder, unique_entities)


# train_encoder(args)
evaluate_test(args, Corpus_.unique_entities_train)
