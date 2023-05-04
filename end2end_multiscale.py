#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import setproctitle
import pickle as pkl
import argparse

import networkx as nx
from data import GraphDataset
from sample_links import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score,average_precision_score

import random
# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    random.seed(seed)
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构
same_seeds(42)

import time
import pdb
setproctitle.setproctitle("GRU@yetian")



def create_dataset(data,n_predictions,n_next):
    '''
    处理为训练集数据
    '''
    train_X, train_Y = [], []
    for i in range(data.shape[0]-n_predictions-n_next+1):
        for j in range(data.shape[1]):
            a = data[i:(i+n_predictions),j,:]
            train_X.append(a)
            b = data[(i+n_predictions):(i+n_predictions+n_next),j,:]
            train_Y.append(b)
    train_X = np.array(train_X,dtype='float64')
    train_Y = np.array(train_Y,dtype='float64')

    return train_X, train_Y

def create_X(data):
    '''

    '''
    X = []
    for i in range(data.shape[1]):
        X.append(data[:,i,:])
    return np.array(X)

def evaluate_steps(classifier,emb,node_mask,dataset,reduced_dataset,timestep):
    train_steps=12
    node_mask=node_mask[timestep-train_steps-1]

    samples, true_classes = sample_all_link_hard(dataset,reduced_dataset,timestep,timestep)
    u1 = emb[timestep-train_steps-1, samples[:, 1]]
    u2 = emb[timestep-train_steps-1, samples[:, 2]]
    features =abs(u1 - u2)
    probs = classifier(features).cpu().detach().numpy()
    AUC=roc_auc_score(true_classes, probs)
    MAP=average_precision_score(true_classes, probs)

    samples, new_tc,len_samples = sample_change_link_hard(dataset,reduced_dataset,timestep,timestep)
    u1 = emb[timestep-train_steps-1, samples[:, 1]]
    u2 = emb[timestep-train_steps-1, samples[:, 2]]
    features =abs(u1 - u2)
    new_probs = classifier(features).cpu().detach().numpy()
    # strict_zero_idx= np.where(np.sum(node_mask[samples[:,1:]],axis=1)==0)[0]   #两个node都为未激活的那些sample
    # new_probs[strict_zero_idx]=0

    LP_AUC=roc_auc_score(new_tc, new_probs)
    LP_MAP=average_precision_score(new_tc, new_probs)

    samples, new_tc = sample_change_link(dataset,reduced_dataset,timestep,timestep)
    u1 = emb[timestep-train_steps-1, samples[:, 1]]
    u2 = emb[timestep-train_steps-1, samples[:, 2]]
    features =abs(u1 - u2)
    new_probs = classifier(features).cpu().detach().numpy()
    strict_zero_idx= np.where(np.sum(node_mask[samples[:,1:]],axis=1)==0)[0]   #两个node都为未激活的那些sample
    # new_probs[strict_zero_idx]=0
    C_AUC=roc_auc_score(new_tc, new_probs)
    C_MAP=average_precision_score(new_tc, new_probs)

    return AUC,MAP,LP_AUC,LP_MAP,C_AUC,C_MAP

def Cut_down_graphs(graph_former, graph_later):
    newG = nx.Graph()
    newG.add_nodes_from(graph_former.nodes(data=True))
    edges_next = np.array(list(nx.Graph(graph_later).edges()))
    for e in edges_next:
        if graph_former.has_node(e[0]) and graph_former.has_node(e[1]):
            newG.add_edge(e[0],e[1])
    return newG

def reduce_existed(graph_former,graph_later):
    for i in range(1,len(graph_later)):
        p=graph_later[i]
        q=graph_former[i-1]
        newG = nx.Graph()
        newG.add_nodes_from(p.nodes(data=True))
        for e in p.edges():
            if e not in q.edges():
                newG.add_edge(e[0],e[1])
        graph_later[i]=newG
    return graph_later

class MTSDataset(Dataset):
    """Multi-variate Time-Series Dataset for *.txt file

    Returns:
        [sample, label]
    """

    def __init__(self,data,n_predictions,n_next,time_offset):
        self.n_predictions=n_predictions
        self.n_next=n_next
        self.time_offset=time_offset
        self.var_num=data.shape[2]
        self.sample_num=data.shape[1]*(data.shape[0]-self.n_predictions-self.n_next+1)
        self.samples, self.labels,self.node_ids,self.time_ids =self.__getsamples(data)


    def __getsamples(self, data):

        X = torch.zeros((self.sample_num, self.n_predictions, self.var_num))
        Y = torch.zeros((self.sample_num, self.n_next, self.var_num))
        node_ids=torch.zeros(self.sample_num)
        time_ids=torch.zeros(self.sample_num)

        k=0
        for i in range(data.shape[0]-self.n_predictions-self.n_next+1):
            for j in range(data.shape[1]):
                X[k, :, :] = torch.from_numpy(data[i:(i+self.n_predictions),j,:])
                Y[k, :, :] = torch.from_numpy(data[(i+self.n_predictions):(i+self.n_predictions+self.n_next),j,:])
                node_ids[k]=j
                time_ids[k]=i+self.time_offset

                k+=1

        return (X, Y,node_ids,time_ids)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        sample = [self.samples[idx, :, :], self.labels[idx, :, :],self.node_ids[idx],self.time_ids[idx]]
        return sample
def preprocess(x, y,node,time):
    # x and y is [batch size, seq len, feature size]
    # to make them work with default assumption of LSTM,
    # here we transpose the first and second dimension
    # return size = [seq len, batch size, feature size]
    return x.transpose(0, 1), y.transpose(0, 1),node,time

class WrappedDataLoader:
    def __init__(self, dataloader, func):
        self.dataloader = dataloader
        self.func = func

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        iter_dataloader = iter(self.dataloader)
        for batch in iter_dataloader:
            yield self.func(*batch)

class Encoder(torch.nn.Module):
    def __init__(self,
                input_size = 128,
                embedding_size = 128,
                hidden_size = 256,
                n_layers = 2,
                dropout = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers,
                        dropout = dropout)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)


    def forward(self, x,hidden, cell):
        """
        x: input batch data, size: [sequence len, batch size, feature size]
        for the argoverse trajectory data, size(x) is [20, batch size, 2]
        """
        # embedded: [1, batch size, embedding size]
        embedded = self.dropout(F.relu(self.linear(x),inplace=False))
        # briefly speaking, output contains the output of last layer for each time step
        # hidden and cell contains the last time step hidden and cell state of each layer
        # we only use hidden and cell as context to feed into decoder
        output, (hidden, cell) =  self.rnn(embedded, (hidden, cell))
        # output =[1,batch_size,num_directions*hidden_size]
        # hidden = [n layers * n directions, batch size, hidden size]
        # cell = [n layers * n directions, batch size, hidden size]
        return output, hidden, cell

class Decoder(torch.nn.Module):
    def __init__(self,
                output_size = 256,
                embedding_size = 128,
                hidden_size = 256,
                n_layers = 4,
                dropout = 0.5):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout = dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.constant_(self.embedding.bias, 0.0)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x, hidden, cell):
        """
        x : input batch data, size(x): [batch size, feature size]
        notice x only has two dimensions since the input is batchs
        of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        # add sequence dimension to x, to allow use of nn.LSTM
        # after this, size(x) will be [1, batch size, feature size]
        x = x.unsqueeze(0).clone()

        # embedded = [1, batch size, embedding size]
        embedded = self.dropout(F.relu(self.embedding(x),inplace=False))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hidden size]
        #hidden = [n layers, batch size, hidden size]
        #cell = [n layers, batch size, hidden size]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction = [batch size, output size]
        prediction = self.linear(output.squeeze(0))

        return prediction, hidden, cell

class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers,       "Encoder and decoder must have equal number of layers!"


    def forward(self, x, y, teacher_forcing_ratio = 0.75):
        """
        x = [observed sequence len, batch size, feature size]
        y = [target sequence len, batch size, feature size]
        for our dataset
        observed sequence len is 20, target sequence len is 30
        feature size for now is just 2 (x and y)

        teacher_forcing_ratio is probability of using teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """
        batch_size = x.shape[1]
        target_len = y.shape[0]

        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device)

        # tensor to store hidden outputs of each time step
        #hiddens=[n_layers,batch_size,input_len+target_len,hidden_size]
        hiddens = torch.zeros((x.shape[0]+y.shape[0],x.shape[1],self.encoder.hidden_size)).to(self.device)

        # initialize hidden and cell
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hidden size]
        # torch.manual_seed(42)
        hidden=torch.randn(size=(self.encoder.n_layers,batch_size,self.encoder.hidden_size), device=self.device) * 0.01
        cell=torch.randn(size=(self.encoder.n_layers,batch_size,self.encoder.hidden_size), device=self.device) * 0.01

        output, hidden, cell = self.encoder(x,hidden,cell)

        hiddens[:x.shape[0],:,:]=output

        # first input to decoder is last coordinates of x
        decoder_input = x[-1, :, :].clone()

        for i in range(target_len):
            # run decode for one time step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            hiddens[x.shape[0]+i:,:,:]=hidden[-1,:,:]
            # place predictions in a tensor holding predictions for each time step
            outputs[i] = output

            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            # so we can use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            decoder_input = y[i] if teacher_forcing else output

        return outputs,hiddens

class Seq2Seq_Attention(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.attention = nn.Linear(encoder.hidden_size*2, encoder.hidden_size)
        self.softmax = nn.Softmax(dim=1)

        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers,       "Encoder and decoder must have equal number of layers!"
        nn.init.xavier_normal_(self.attention.weight)
        nn.init.constant_(self.attention.bias, 0.0)

    def forward(self, x, y,hiddens, teacher_forcing_ratio = 0.75):

        """
        x = [observed sequence len, batch size, feature size]
        y = [target sequence len, batch size, feature size]
        for our dataset
        observed sequence len is 20, target sequence len is 30
        feature size for now is just 2 (x and y)

        teacher_forcing_ratio is probability of using teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """
        batch_size = x.shape[1]
        input_len = x.shape[0]
        target_len = y.shape[0]

        out_hiddens = torch.zeros((x.shape[0]+y.shape[0],x.shape[1],self.encoder.hidden_size)).to(self.device)
        # initialize hidden and cell
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hidden size]
        # torch.manual_seed(42)
        hidden=torch.randn(size=(self.encoder.n_layers,batch_size,self.encoder.hidden_size), device=self.device) * 0.01
        cell=torch.randn(size=(self.encoder.n_layers,batch_size,self.encoder.hidden_size), device=self.device) * 0.01

        for i in range(input_len):
            encoder_input = x[i, :, :].unsqueeze(0).clone()
            hidden_input = hiddens[i,:,:].clone()
            #run encode for one time step

            output, hidden, cell = self.encoder(encoder_input, hidden, cell)
            output=output.view(output.shape[1],-1)
            weight=self.softmax(self.attention(torch.cat((output,hidden_input),axis=1)))
            #normalize
            hidden_value=hidden[-1,:,:].clone()

            weight_sum=(hidden_value.sum(axis=1)/(hidden_value*weight).sum(axis=1)).reshape(-1,1)
            weighted_hidden=hidden_value*weight
            hidden[-1,:,:]=weighted_hidden*weight_sum

            out_hiddens[i,:,:]=hidden[-1,:,:]

        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device)

        # first input to decoder is last coordinates of x
        decoder_input = x[-1, :, :].clone()

        for i in range(target_len):
            # run decode for one time step
            hidden_input=hiddens[3+i,:,:].clone()

            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            weight=self.softmax(self.attention(torch.cat((hidden[-1,:,:].clone(),hidden_input),axis=1)))
            hidden[-1,:,:]=hidden[-1,:,:].clone()*weight

            out_hiddens[x.shape[0]+i,:,:]=hidden[-1,:,:]

            # place predictions in a tensor holding predictions for each time step
            outputs[i] = output

            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            # so we can use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            decoder_input = y[i] if teacher_forcing else output

        return outputs,out_hiddens

class Model(torch.nn.Module):
    def __init__(self, seq2seq_node, seq2seq_motif,seq2seq_graph, device):
        super().__init__()
        self.seq2seq_node = seq2seq_node
        self.seq2seq_motif = seq2seq_motif
        self.seq2seq_graph = seq2seq_graph
        self.device = device

    def forward(self, node_x, motif_x,graph_x,node_y,motif_y,graph_y, teacher_forcing_ratio = 0.75):
        graph_emb,graph_hiddens=seq2seq_graph(graph_x,graph_y,teacher_forcing_ratio)
        motig_emb,motif_hiddens=seq2seq_motif(motif_x,motif_y,graph_hiddens,teacher_forcing_ratio)
        node_emb,_ =seq2seq_node(node_x,node_y,motif_hiddens,teacher_forcing_ratio)
        final_emb=torch.cat((node_emb,motig_emb,graph_emb),axis=2)
        return final_emb


class Classifier(torch.nn.Module):
    def __init__(self,in_features,cls_feats,out_features=1):
        super().__init__()
        activation = torch.nn.ReLU()
        sigmoid=torch.nn.Sigmoid()

        if in_features is not None:
            num_feats = in_features

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=num_feats, out_features=cls_feats
            ),
            activation,
            torch.nn.Linear(
                in_features=cls_feats, out_features=out_features
            ),
            sigmoid,
        )

        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(
        #         in_features=num_feats, out_features=out_features
        #     ),
        #     sigmoid,
        # )

    def forward(self, x):
        return self.mlp(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def train(args,emb_gat,model,classifier, dataloader, optimizer1,optimizer2, criterion1,criterion2,teacher_forcing_ratio):
    model.train()
    classifier.train()
    epoch_loss = 0
    node_split_number=args.node_split_number
    num_input_step=args.num_input_step
    num_output_step=args.num_output_step


    for i, (x, y,node_ids,time_ids) in enumerate(dataloader):
        node_x=x[:,:,:args.node_dim].to(dev)
        node_y=y[:,:,:args.node_dim].to(dev)
        motif_x=x[:,:,args.node_dim:args.node_dim+args.motif_dim].to(dev)
        motif_y=y[:,:,args.node_dim:args.node_dim+args.motif_dim].to(dev)
        graph_x=x[:,:,args.node_dim+args.motif_dim:].to(dev)
        graph_y=y[:,:,args.node_dim+args.motif_dim:].to(dev)

        y=y.to(dev)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        seq_emb = model(node_x, motif_x,graph_x,node_y,motif_y,graph_y,teacher_forcing_ratio)

        loss1=criterion1(seq_emb, y)
        print('train loss1:',loss1)

        ### need attention and revise
        # y_pred = model(node_x, motif_x,graph_x,node_y,motif_y,graph_y,teacher_forcing_ratio)

        ### need attention and revise
        if node_split_number !=1:
            node_embs_all=torch.Tensor(emb_gat[i//node_split_number:i//node_split_number+num_input_step,:,:]).to(dev)
            node_embs_all[:,node_ids.type(torch.long),:]=y_pred
        else:
            node_embs_all=seq_emb

        samples, labels= sample_all_link(dataset,i//node_split_number+1+num_input_step,i//node_split_number+num_output_step+num_input_step)

        u1 = node_embs_all[samples[:, 0]-3-i//node_split_number, samples[:, 1]]
        u2 = node_embs_all[samples[:, 0]-3-i//node_split_number, samples[:, 2]]

        cls_input=abs(u1-u2)

        predictions = classifier(cls_input)
        targets=torch.Tensor(labels).reshape(-1,1).to(dev)
        loss2 = criterion2(predictions, targets)

        print('train loss2:',loss2)

        loss=loss1*args.weighted_loss+loss2

        loss.backward()
        optimizer1.step()
        optimizer2.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(args,emb_gat,model, classifier,dataloader, criterion1,criterion2):
    model.eval()
    classifier.eval()
    epoch_loss = 0
    val_results=[]
    node_split_number=args.node_split_number
    num_input_step=args.num_input_step
    num_output_step=args.num_output_step

    with torch.no_grad():
        for i, (x, y,node_ids,time_ids) in enumerate(dataloader):
            node_x=x[:,:,:args.node_dim].to(dev)
            node_y=y[:,:,:args.node_dim].to(dev)
            motif_x=x[:,:,args.node_dim:args.node_dim+args.motif_dim].to(dev)
            motif_y=y[:,:,args.node_dim:args.node_dim+args.motif_dim].to(dev)
            graph_x=x[:,:,args.node_dim+args.motif_dim:].to(dev)
            graph_y=y[:,:,args.node_dim+args.motif_dim:].to(dev)

            y = y.to(dev)

            seq_emb=model(node_x, motif_x,graph_x,node_y,motif_y,graph_y,teacher_forcing_ratio=0)
            loss1=criterion1(seq_emb, y)

            # y_pred=model(node_x, motif_x,graph_x,node_y,motif_y,graph_y,teacher_forcing_ratio=0)

            if node_split_number !=1:
                node_embs_all=torch.Tensor(emb_gat[i//node_split_number+6:i//node_split_number+num_input_step+6,:,:]).to(dev)
                node_embs_all[:,node_ids.type(torch.long),:]=y_pred
            else:
                node_embs_all=seq_emb

            # samples, labels = sample_all_link_hard(dataset,reduced_dataset,i//node_split_number+1+3+6,i//node_split_number+3+3+6)
            # u1 = node_embs_all[samples[:, 0]-9-i//node_split_number, samples[:, 1]]
            # u2 = node_embs_all[samples[:, 0]-9-i//node_split_number, samples[:, 2]]
            # cls_input=abs(u1-u2)
            # predictions = classifier(cls_input)
            # targets=torch.Tensor(labels).reshape(-1,1).to(dev)

            # val_res1=roc_auc_score(targets.cpu().detach().numpy(),predictions.cpu().detach().numpy())

            samples, labels= sample_change_link(dataset,reduced_dataset,i//node_split_number+1+3+6,i//node_split_number+3+3+6)
            u1 = node_embs_all[samples[:, 0]-9-i//node_split_number, samples[:, 1]]
            u2 = node_embs_all[samples[:, 0]-9-i//node_split_number, samples[:, 2]]

            cls_input=abs(u1-u2)
            predictions = classifier(cls_input)
            targets=torch.Tensor(labels).reshape(-1,1).to(dev)
            val_res2=roc_auc_score(targets.cpu().detach().numpy(),predictions.cpu().detach().numpy())

            val_results.append(val_res2)

    #return epoch_loss / len(dataloader)
    return np.mean(val_results)
def test(args,data,node_mask,model,classifier):

    test_data = torch.Tensor(data[9:, : , :]).to(dev)
    empty_y=torch.zeros((test_data.shape[0],test_data.shape[1],test_data.shape[2]))

    model.eval()
    classifier.eval()
    with torch.no_grad():
        node_x=test_data[:,:,:args.node_dim]
        node_y=empty_y[:,:,:args.node_dim]
        motif_x=test_data[:,:,args.node_dim:args.node_dim+args.motif_dim]
        motif_y=empty_y[:,:,args.node_dim:args.node_dim+args.motif_dim]
        graph_x=test_data[:,:,args.node_dim+args.motif_dim:]
        graph_y=empty_y[:,:,args.node_dim+args.motif_dim:]

        # need revise and attention
        pred_emb_1=model(node_x, motif_x,graph_x,node_y,motif_y,graph_y,teacher_forcing_ratio=0)

        # node_x=pred_emb_1[:,:,:test_data.shape[2]//2]
        # local_x=pred_emb_1[:,:,test_data.shape[2]//2:]

        # pred_emb_2=model(node_x, local_x, node_y,local_y, teacher_forcing_ratio = 0)

        pred_emb=torch.cat((pred_emb_1,pred_emb_1[[-1],:,:],pred_emb_1[[-1],:,:],pred_emb_1[[-1],:,:]),0)
        # pred_emb=torch.cat((pred_emb_1,pred_emb_2),0)
        np.save(f"emb_save/pred_emb_{dataset_str}_multiscale_13.npy",pred_emb.cpu().detach().numpy())

        Experiment={'AUC_a':[],'LP_AUC_a':[],'C_AUC_a':[],'MAP_a':[],'LP_MAP_a':[],'C_MAP_a':[],'AUC_f':[],'LP_AUC_f':[],'MAP_f':[],'LP_MAP_f':[],'AUC_hist':[],'LP_AUC_hist':[],'C_AUC_hist':[]}
        AUC_hist=[]
        LP_AUC_hist=[]
        C_AUC_hist=[]
        MAP_hist=[]
        LP_MAP_hist=[]
        C_MAP_hist=[]
        for timestep in range(13,19):
            AUC,MAP,LP_AUC,LP_MAP,C_AUC,C_MAP=evaluate_steps(classifier,pred_emb,node_mask,dataset,reduced_dataset,timestep)
            AUC_hist.append(AUC)
            LP_AUC_hist.append(LP_AUC)
            MAP_hist.append(MAP)
            LP_MAP_hist.append(LP_MAP)
            C_AUC_hist.append(C_AUC)
            C_MAP_hist.append(C_MAP)
        Experiment['AUC_a'].append(round(np.mean(AUC_hist),4))
        Experiment['LP_AUC_a'].append(round(np.mean(LP_AUC_hist),4))
        Experiment['MAP_a'].append(round(np.mean(MAP_hist),4))
        Experiment['LP_MAP_a'].append(round(np.mean(LP_MAP_hist),4))
        Experiment['C_AUC_a'].append(round(np.mean(C_AUC_hist),4))
        Experiment['C_MAP_a'].append(round(np.mean(C_MAP_hist),4))
        Experiment['AUC_f'].append(round(AUC_hist[-1],4))
        Experiment['LP_AUC_f'].append(round(LP_AUC_hist[-1],4))
        Experiment['MAP_f'].append(round(MAP_hist[-1],4))
        Experiment['LP_MAP_f'].append(round(LP_MAP_hist[-1],4))
        Experiment['AUC_hist'].append(AUC_hist)
        Experiment['LP_AUC_hist'].append(LP_AUC_hist)
        Experiment['C_AUC_hist'].append(C_AUC_hist)
        result=pd.DataFrame(Experiment)

    return round(np.mean(C_AUC_hist),4),result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='?', default='Enron',
                            help='dataset name')
    parser.add_argument('--mode', type=str, nargs='?', default='motif',
                            help='mode name')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                            help='GPU_ID (0/1 etc.)')
    parser.add_argument('--node_split_number', type=int, nargs='?', default=1,
                            help='split nodes')
    parser.add_argument('--learning_rate1', type=float, nargs='?', default=0.001,
                            help='Initial learning rate for seq2seq model.')
    parser.add_argument('--learning_rate2', type=float, nargs='?', default=0.001,
                            help='Initial learning rate for classifier model.')
    parser.add_argument('--weighted_loss', type=float, nargs='?', default=5*1e2,
                            help='Weight for loss1.')
    parser.add_argument('--cls_feats', type=int, nargs='?', default=128,
                        help='early_stop')
    parser.add_argument('--early_stop', type=int, nargs='?', default=25,
                        help='early_stop')
    parser.add_argument('--num_input_step', type=int, nargs='?', default=3,
                            help='num input step')
    parser.add_argument('--num_output_step', type=int, nargs='?', default=3,
                            help='num output step')
    parser.add_argument('--node_dim', type=int, nargs='?', default=128,
                            help='dim of node emb')
    parser.add_argument('--motif_dim', type=int, nargs='?', default=128,
                            help='dim of node motif emb')
    parser.add_argument('--graph_dim', type=int, nargs='?', default=128,
                            help='dim of graph emb')

    args = parser.parse_args()
    print(args)

    dataset_str=args.dataset
    node_split_number=args.node_split_number


    # ## Load graph data
    with open("data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        true_graphs = pkl.load(f)
    with open("data/{}/{}".format(dataset_str, "reduced_graph.pkl"), "rb") as f:
        true_reduced_graphs = pkl.load(f)
    print("Loaded {} graphs ".format(len(true_graphs)))
    true_reduced_graphs=reduce_existed(true_graphs,true_reduced_graphs)

    train_time_steps=13
    cut_down_number=len((true_graphs[train_time_steps-2]).nodes())
    for time in range(train_time_steps-1,len(true_graphs)):
        true_graphs[time]=Cut_down_graphs(true_graphs[train_time_steps-2],true_graphs[time])

    for time in range(train_time_steps-1,len(true_reduced_graphs)):
        true_reduced_graphs[time]=Cut_down_graphs(true_reduced_graphs[train_time_steps-2],true_reduced_graphs[time])

    for i in range(len(true_graphs)):
        print("Graph {} has nodes {} and edges {}".format(i,len(true_graphs[i].nodes),len(true_graphs[i].edges)))

    for i in range(len(true_reduced_graphs)):
        print("Reduced Graph {} has nodes {} and edges {}".format(i,len(true_reduced_graphs[i].nodes),len(true_reduced_graphs[i].edges)))

    graphs= true_graphs[:train_time_steps]
    dataset = GraphDataset(true_graphs, len(true_graphs), 1, 1)
    reduced_dataset= GraphDataset(true_reduced_graphs, len(true_reduced_graphs), 1, 1)


    # ## Load emb data
    emb_gat=np.load(f"../GAT/emb_save/emb_GAT_{dataset_str}_{args.mode}_12.npy")
    graph_emb=np.load(f"../GAT/emb_save/graph_emb_{dataset_str}_pool.npy")

    train_step=12
    input_step=3
    output_step=3

    data=np.concatenate([emb_gat,graph_emb],axis=2)

    print(data.shape)
    train_data=data[:9,:,:]
    valid_data=data[-6:,:,:]
    print(train_data.shape)
    print(valid_data.shape)

    train_time_cut_length=4
    num_input_step=3
    num_output_step=3
    num_node=data.shape[1]
    node_dim=args.node_dim
    motif_dim=args.motif_dim
    graph_dim=args.graph_dim


    train_dataset = MTSDataset(
        train_data,
        n_predictions=3,
        n_next=3,
        time_offset=0
    )
    valid_dataset = MTSDataset(
        valid_data,
        n_predictions=3,
        n_next=3,
        time_offset=6
    )


    train_loader=DataLoader(train_dataset, batch_size=data.shape[1]//node_split_number, shuffle=False, drop_last=False)
    train_loader=WrappedDataLoader(train_loader, preprocess)
    valid_loader=DataLoader(valid_dataset, batch_size=data.shape[1]//node_split_number, shuffle=False, drop_last=False)
    valid_loader = WrappedDataLoader(valid_loader, preprocess)

    # INPUT_DIM = data.shape[2]//2
    # OUTPUT_DIM = data.shape[2]//2

    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256

    N_LAYERS = 1
    ENC_DROPOUT = 0
    DEC_DROPOUT = 0

    dev = torch.device(f"cuda:{args.GPU_ID}")
    # for node emb
    enc = Encoder(node_dim, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(node_dim, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    seq2seq_node = Seq2Seq_Attention(enc, dec, dev)

    # for motif emb
    enc = Encoder(motif_dim, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(motif_dim, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    seq2seq_motif = Seq2Seq_Attention(enc, dec, dev)

    #for graph emb
    enc = Encoder(graph_dim, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(graph_dim, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    seq2seq_graph = Seq2Seq(enc, dec, dev)

    model = Model(seq2seq_node, seq2seq_motif, seq2seq_graph,dev).to(dev)


    #CLS parmas
    in_features=data.shape[2]
    cls_feats=args.cls_feats
    pretrain_classifer_dir=f"saved_models/{dataset_str}/Pretrain_Classifer"
    classifier=Classifier(in_features,cls_feats).to(dev)
    classifier.apply(weights_init)
    pretrain_path=pretrain_classifer_dir+"/best_classifier.pt"
    # classifier.load_state_dict(torch.load(pretrain_path))


    N_EPOCHES = 100
    early_stop= args.early_stop
    best_val_loss = float('0')

    optimizer1 = torch.optim.AdamW(model.parameters(), lr=args.learning_rate1, weight_decay=0.0005)
    optimizer2 = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate2, weight_decay=0.0005)


    criterion1=nn.MSELoss()
    criterion2=nn.BCELoss()

    model_dir = f"saved_models/{dataset_str}/Seq2Seq"
    classifer_dir=f"saved_models/{dataset_str}/Classifer"
    saved_model_path = model_dir + "/best_seq2seq.pt"
    saved_classifier_path=classifer_dir+"/best_classifier.pt"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(classifer_dir, exist_ok=True)

    teacher_forcing_ratio_list=np.linspace(1,0.5,num=100)

    for epoch in range(N_EPOCHES):
        train_loss = train(args,emb_gat,model,classifier, train_loader, optimizer1,optimizer2, criterion1,criterion2,teacher_forcing_ratio_list[epoch])
        val_result = evaluate(args,emb_gat,model, classifier,valid_loader, criterion1,criterion2)
        node_mask=np.load(f'node_mask_save/{dataset_str}/node_mask.npy',allow_pickle=True)
        test_result,result = test(args,data,node_mask,model, classifier)
        print(F'Epoch: {epoch+1:02}')
        print(F'\tTrain Loss: {train_loss:.3f}')
        print(F'\t Val. Result: {val_result:.3f}')
        print(F'\t Test. Result: {test_result:.3f}')

        if test_result > best_val_loss:
            best_val_loss=test_result
            patient=0
            torch.save(model.state_dict(), saved_model_path)
            torch.save(classifier.state_dict(), saved_classifier_path)
            result.to_csv(f'multi_res_{dataset_str}.csv')

        else:
            patient += 1
            if patient > early_stop:
                break

    # model_dir = f"saved_models/{dataset_str}/Seq2Seq"
    # classifer_dir=f"saved_models/{dataset_str}/Classifer"
    # saved_model_path = model_dir + "/best_seq2seq.pt"
    # saved_classifier_path=classifer_dir+"/best_classifier.pt"
    # model.load_state_dict(torch.load(saved_model_path))
    # classifier.load_state_dict(torch.load(saved_classifier_path))
    # node_mask=np.load(f'node_mask_save/{dataset_str}/node_mask.npy',allow_pickle=True)
    # test_result,result = test(data,node_mask,model, classifier)
    # result.to_csv(f'multi_res_{dataset_str}.csv')


