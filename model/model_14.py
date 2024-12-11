import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Param:
        in_features, out_features, bias
    Input:
        features: N x C (n = # nodes), C = in_features
        adj: adjacency matrix (N x N)
    """

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
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT Layer)
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(1, out_features * 2))  # Attention weights

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)  # Softmax over neighbors

    def forward(self, h, adj):
        # h: [B, N, F'] -> Node features
        B, N, F = h.size()

        # Linear transformation
        h_prime = self.W(h)  # [B, N, F'']

        # Compute attention coefficients (matrix multiplication to avoid redundant concatenation)
        e = torch.matmul(h_prime, h_prime.transpose(1, 2))  # [B, N, N]
        e = self.leakyrelu(e)  # [B, N, N]
        attention = self.softmax(e)  # Softmax across rows [B, N, N]

        # Apply attention
        h_prime = h_prime.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, F'']
        h_prime = h_prime * attention.unsqueeze(-1)  # [B, N, N, F'']

        # Aggregate information from neighbors
        output = torch.sum(h_prime, dim=2)  # [B, N, F'']

        return output


class GCNWithGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCNWithGAT, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
        self.gat1 = GraphAttentionLayer(nhid, nout, dropout)

        self.bn1 = nn.BatchNorm1d(nhid)

    def forward(self, x, adj):
        x = self.gc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        x = F.relu(x)
        # Pass adj to GAT layer
        x = self.gat1(x, adj)
        return x


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilated_conv = nn.Conv1d(input_size if i == 0 else num_channels[i-1], num_channels[i],
                                     kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=2**i)
            layers.append(dilated_conv)
            layers.append(nn.BatchNorm1d(num_channels[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AUwGCNWithGAT(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        mat_dir = '/kaggle/working/ME-GCN-Project'
        self.mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # Graph Embedding Module
        self.graph_embedding = GCNWithGAT(2, 16, 16, self.mat_path)

        in_dim = 192  # Output from GCN and GAT
        # Temporal Convolutional Network
        self.temporal_model = TemporalConvNet(input_size=in_dim, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2)

        # Global pooling
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        self._classification = torch.nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # Assume 10 classes, adjust accordingly
        )

        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # Flatten time dimension (b*t, n, c)

        # Get adjacency matrix adj
        adj = self.graph_embedding.gc1.adj  # Get adj from graph_embedding
        # Perform GCN and GAT operations
        x = self.graph_embedding(x, adj)

        # Reshape for TCN input
        x = x.reshape(b, t, -1).transpose(1, 2)  # [b, c, t]

        # Apply TCN
        x = self.temporal_model(x)  # [b, c, t]

        # Global pooling
        x = self.global_pooling(x)  # [b, c, 1]

        # Flatten and classify
        x = x.squeeze(-1)  # [b, c]
        x = self._classification(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Parameter):
                m.data.uniform_(-0.1, 0.1)  # Adjust based on requirements
