import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np


# Graph Convolution Layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, mat_path, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # 直接在初始化时加载邻接矩阵
        adj_mat = np.load(mat_path)
        self.register_buffer('adj', torch.from_numpy(adj_mat))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, c = input.shape
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        output = torch.bmm(self.adj.unsqueeze(0).repeat(b, 1, 1), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# Graph Attention Layer (GAT)
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(1, out_features * 2))  # 注意力权重

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)  # Softmax 是对每个节点的邻居节点计算的

    def forward(self, h, adj):
        # h: [B, N, F'] -> 节点特征
        B, N, F = h.size()

        # 将输入特征进行线性变换
        h_prime = self.W(h)  # [B, N, F'']

        # 计算注意力系数（改为矩阵乘法，避免重复拼接）
        e = torch.matmul(h_prime, h_prime.transpose(1, 2))  # [B, N, N]
        e = self.leakyrelu(e)  # [B, N, N]
        attention = self.softmax(e)  # 对行进行 softmax [B, N, N]

        # 应用注意力机制
        h_prime = h_prime.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, F'']
        h_prime = h_prime * attention.unsqueeze(-1)  # [B, N, N, F'']

        # 聚合邻居信息
        output = torch.sum(h_prime, dim=2)  # [B, N, F'']

        return output


# Temporal Convolutional Block (TCN)
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * dilation // 2, dilation=dilation
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


# GCN with GAT and TCN Integration
class GCNWithGATAndTCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCNWithGATAndTCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
        self.gat1 = GraphAttentionLayer(nhid, nout, dropout)
        self.tcn1 = TCNBlock(nout, nout, kernel_size=3, dilation=1, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(nhid)

        # Residual connection adjustment
        self.residual = nn.Conv1d(nfeat, nout, kernel_size=1, bias=False)

    def forward(self, x, adj):
        residual = self.residual(x.transpose(1, 2)).transpose(1, 2)
        x = self.gc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.gat1(x, adj)
        x = self.tcn1(x.transpose(1, 2)).transpose(1, 2)
        x += residual  # Residual connection
        return x


# Final Model with AUwGCN, GAT, TCN
class AUwGCNWithGATAndTCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        mat_dir = '/kaggle/working/ME-GCN-Project'
        self.mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        self.graph_embedding = GCNWithGATAndTCN(2, 16, 16, self.mat_path)

        in_dim = 192
        self._sequential = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            # Additional convolution layers
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )

        # Temporal attention module
        self.attention = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.Softmax(dim=2)
        )

        # Final classification layer
        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False
        )

        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape
        x = x.reshape(b * t, n, c)
        adj = self.graph_embedding.gc1.adj

        x = self.graph_embedding(x, adj)
        x = x.reshape(b, t, -1).transpose(1, 2)
        x = self._sequential(x)

        # Temporal attention weighting
        attention_weights = self.attention(x)
        x = x * attention_weights

        x = self._classification(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Parameter):
                m.data.uniform_(-0.1, 0.1)