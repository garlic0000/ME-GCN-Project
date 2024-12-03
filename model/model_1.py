import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
关键更改：
GraphAttentionLayer：

添加了一个 GraphAttentionLayer 类，它实现了一个基本的图注意力机制（类似于 GAT）。它的核心思想是计算每个节点和其邻居之间的注意力权重，然后加权求和。
W 用于节点特征的线性变换，a 是用于计算节点对之间注意力分数的参数。
在 AUwGCN 中使用注意力层：

AUwGCN 中添加了 GraphAttentionLayer，在 graph_embedding 后使用它来增强图卷积层的表示能力。
去除 weight_norm：GraphConvolution 层已恢复为普通的初始化方式，没有使用 weight_norm。

注意力机制：通过图注意力层 (GraphAttentionLayer)，为每个节点和其邻居分配不同的权重，这样模型能够关注重要的节点关系。
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
        self.weight = Parameter(torch.Tensor(in_features, out_features))  # no weight_norm
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        adj_mat = np.load(mat_path)
        self.register_buffer('adj', torch.from_numpy(adj_mat))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, c = input.shape
        # Apply weight to the input
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))  # Shape: [B, N, F] x [F, O]
        output = torch.bmm(self.adj.unsqueeze(0).repeat(b, 1, 1), support)  # Shape: [B, N, N] x [B, N, O]

        if self.bias is not None:
            return output + self.bias  # Shape: [B, N, O]
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
        self.bn1 = nn.BatchNorm1d(nhid)

    def forward(self, x):
        x = self.gc1(x)
        x = x.transpose(1, 2).contiguous()
        x = self.bn1(x).transpose(1, 2).contiguous()
        x = F.relu(x)

        return x, self.gc1.adj  # 返回图的邻接矩阵 adj


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))  # Attention parameters
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # Linear transformation
        # Wh: [B, N, F]

        # Pairwise attention
        # Repeat Wh to match pairwise combinations
        Wh_repeat_1 = Wh.unsqueeze(2).repeat(1, 1, Wh.size(1), 1)  # Shape: [B, N, N, F]
        Wh_repeat_2 = Wh.unsqueeze(1).repeat(1, Wh.size(1), 1, 1)  # Shape: [B, N, N, F]
        a_input = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=-1)  # Shape: [B, N, N, 2F]

        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # Attention scores
        attention = torch.nn.functional.softmax(e, dim=1)  # Shape of attention: [B, N, N]

        # Reshape Wh from [B, N, F] to [B, N, F] for compatibility with attention
        Wh_reshaped = Wh  # Shape: [B, N, F]

        # Perform batch matrix multiplication: [B, N, N] * [B, N, F] -> [B, N, F]
        h_prime = torch.bmm(attention, Wh_reshaped)  # Shape: [B, N, F]

        return h_prime


class AUwGCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # 服务器测试路径
        mat_dir = '/kaggle/working/ME-GCN-Project'
        mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, mat_path))

        in_dim = 192  # 24，保留输入通道数为192

        self.attention = GraphAttentionLayer(in_features=16, out_features=16)  # 调整 GraphAttentionLayer

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

        self._classification = torch.nn.Conv1d(64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2,
                                               bias=False)

        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # (b*t, n, c)
        x, adj = self.graph_embedding(x)  # 获取图卷积的输出和邻接矩阵
        x = self.attention(x, adj)  # 将邻接矩阵传递给注意力层

        x = x.reshape(b, t, -1).transpose(1, 2)  # 调整维度
        x = self._sequential(x)
        x = self._classification(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


