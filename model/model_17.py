import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
在model_14基础上增加gat2和tcn2
"""

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
    图注意力层 (GAT Layer)
    """

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


class GCNWithGATAndTCN(nn.Module):
    """
    带有两层 GAT 和 TCN 的模块。
    """
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCNWithGATAndTCN, self).__init__()

        # 第一层 GCN
        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)

        # 第一层 GAT 和 TCN
        self.gat1 = GraphAttentionLayer(nhid, nout, dropout)
        self.tcn1 = TCNBlock(nout, nout, kernel_size=3, dilation=1, dropout=0.2)

        # 第二层 GAT 和 TCN
        self.gat2 = GraphAttentionLayer(nout, nout, dropout)
        self.tcn2 = TCNBlock(nout, nout, kernel_size=3, dilation=2, dropout=0.2)

        # BatchNorm 层
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nout)

    def forward(self, x, adj):
        # 第一层 GCN
        x = self.gc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        x = F.relu(x)

        # 第一层 GAT 和 TCN
        x = self.gat1(x, adj)
        x = self.tcn1(x.transpose(1, 2)).transpose(1, 2)

        # 第二层 GAT 和 TCN
        x = self.gat2(x, adj)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        x = self.tcn2(x.transpose(1, 2)).transpose(1, 2)

        return x


class AUwGCNWithGATAndTCN(torch.nn.Module):
    """
    增强版 AU 检测模型，包含两层 GAT 和 TCN。
    """
    def __init__(self, opt):
        super().__init__()

        mat_dir = '/kaggle/working/ME-GCN-Project'
        self.mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # 使用增强版 GCNWithGATAndTCN
        self.graph_embedding = GCNWithGATAndTCN(2, 16, 16, self.mat_path)

        in_dim = 192  # GCN 和 GAT 输出的维度
        self._sequential = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )

        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False
        )

        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # (b*t, n, c)

        # 获取邻接矩阵 adj
        adj = self.graph_embedding.gc1.adj  # 从 graph_embedding 中获取 adj
        # 调用 GCNWithGATAndTCN 进行图卷积、图注意力和TCN操作
        x = self.graph_embedding(x, adj)
        # reshape 处理为适合卷积输入的维度
        x = x.reshape(b, t, -1).transpose(1, 2)
        # 卷积操作
        x = self._sequential(x)
        # 分类层
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
