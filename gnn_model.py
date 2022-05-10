import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
# from sklearn.preprocessing import normalize
import random
# from transformer.Layers import EncoderLayer ####

from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv


class GIN_Net2(torch.nn.Module):
    def __init__(self, in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1,
                 hidden=512, use_jk=False, pool_size=3, cnn_hidden=1, train_eps=True,
                 feature_fusion=None, class_num=7):
        super(GIN_Net2, self).__init__()
        self.use_jk = use_jk
        self.train_eps = train_eps
        self.feature_fusion = feature_fusion

        self.conv1d = nn.Conv1d(in_channels=in_feature, out_channels=cnn_hidden, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.biGRU = nn.GRU(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=1)
        # self.transformer = EncoderLayer(d_model=666, d_inner=666, n_head=8, d_k=13, d_v=13)  ####
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(in_len / pool_size), gin_in_feature)

        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),  ####
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )  ####

        # self.gin_conv3 = GINConv(
        #     nn.Sequential(
        #         nn.Linear(gin_in_feature, hidden),
        #         nn.ReLU(),
        #         nn.Linear(hidden, hidden),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(hidden),
        #     ), train_eps=self.train_eps
        # )  ####

        self.gin_convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.gin_convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),  ####
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden),
                    ), train_eps=self.train_eps
                )
            )
        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, class_num)

    def reset_parameters(self):

        self.conv1d.reset_parameters()
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        # self.gin_conv2.reset_parameters()  ####
        # self.gin_conv3.reset_parameters() ####
        for gin_conv in self.gin_convs:
            gin_conv.reset_parameters()

        if self.use_jk:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.fc2.reset_parameters()

    def forward(self, x, edge_index, train_edge_id, Gadj2, p=0.5):  ####
        # print(edge_index.dtype)
        # print(Gadj.dtype)
        # exit(0)
        # print(x.shape)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.bn1(x)
        # print(x.shape)
        x = self.maxpool1d(x)
        x = x.transpose(1, 2)
        x, _ = self.biGRU(x)
        # print(x.shape)
        # exit(0)
        # x, _ = self.transformer(x)  ####
        # print(x.shape)
        # exit(0)
        # x = x.transpose(1, 2)  ####
        # print(x.shape)
        # exit(0)
        x = self.global_avgpool1d(x)
        x = x.squeeze()
        x = self.fc1(x)
        # exit(0)
        x1 = self.gin_conv1(x, edge_index)
        # print(x1.shape)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        x2 = self.gin_conv2(x, Gadj2.cuda())  ####
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # # print(x2.shape)
        # x3 = self.gin_conv3(x, Gadj3.cuda()) ####
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        x = x1 + x2 ####

        ####
        xs = [x]
        for conv in self.gin_convs:
            x = conv(x, edge_index)
            xs += [x]
        ####

        if self.use_jk:
            x = self.jump(xs)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=p, training=self.training)
        x = self.lin2(x)
        # x  = torch.add(x, x_)

        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.mul(x1, x2)
        x = self.fc2(x)

        return x