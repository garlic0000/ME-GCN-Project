import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
去掉drop_edge
去掉ResidualWeight
"""



def drop_edge(adj, drop_prob=0.9, epoch=0, max_epochs=100, min_drop_prob=0.09):
    """
    直接返回
    """
    return adj


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, mat_path, bias=True, drop_prob=0.9, min_drop_prob=0.09):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Load adjacency matrix
        adj_mat = np.load(mat_path)
        self.register_buffer('adj', torch.from_numpy(adj_mat))

        # DropEdge probability
        self.drop_prob = drop_prob
        self.min_drop_prob = min_drop_prob

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, epoch=0, max_epochs=100):
        b, n, c = input.shape
        adj = self.adj

        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        output = torch.bmm(adj.unsqueeze(0).repeat(b, 1, 1), support)

        if self.bias is not None:
            output += self.bias

        return F.relu(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiHeadGraphAttentionLayer(nn.Module):
    """
    多头图注意力层 (Multi-Head GAT Layer)
    """

    def __init__(self, in_features, out_features, num_heads=4, dropout=0.6, alpha=0.2):
        super(MultiHeadGraphAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads  # 每个头的输出特征维度
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.ModuleList([nn.Linear(in_features, self.out_per_head, bias=False) for _ in range(num_heads)])
        self.a = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.out_per_head * 2)) for _ in range(num_heads)])

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, h, adj, epoch=0, max_epochs=100):
        B, N, F = h.size()
        outputs = []

        for i in range(self.num_heads):
            h_prime = self.W[i](h)
            e = torch.matmul(h_prime, h_prime.transpose(1, 2))
            e = self.leakyrelu(e)
            attention = self.softmax(e)
            h_prime = h_prime.unsqueeze(2).repeat(1, 1, N, 1)
            h_prime = h_prime * attention.unsqueeze(-1)
            output = torch.sum(h_prime, dim=2)
            outputs.append(output)

        output = torch.cat(outputs, dim=-1)
        return output


class TCNBlock(nn.Module):
    """
    TCN layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=1,
            padding=(kernel_size - 1) * dilation // 2, dilation=dilation
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


class GCNWithMultiHeadGATAndTCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_heads=4):
        super(GCNWithMultiHeadGATAndTCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
        self.gat1 = MultiHeadGraphAttentionLayer(nhid, nout, num_heads, dropout)
        self.tcn1 = TCNBlock(nout, nout, kernel_size=3, dilation=1, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nout)

    def forward(self, x, adj, epoch=0, max_epochs=100):
        x = self.gc1(x, epoch=epoch, max_epochs=max_epochs)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        x = self.gat1(x, adj, epoch=epoch, max_epochs=max_epochs)
        x = self.tcn1(x.transpose(1, 2)).transpose(1, 2)
        return x



class AUwGCNWithMultiHeadGATAndTCN(torch.nn.Module):
    """
    AU detection model with GCN, Multi-Head GAT, and one TCN layer
    """

    def __init__(self, opt):
        super().__init__()

        mat_dir = '/kaggle/working/ME-GCN-Project'
        self.mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # 使用修改后的 GCNWithMultiHeadGATAndTCN
        self.graph_embedding = GCNWithMultiHeadGATAndTCN(2, 16, 16, self.mat_path, num_heads=4)

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

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )

        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False
        )

        self._init_weight()

    def forward(self, x, epoch=0, max_epochs=100):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # (b*t, n, c)

        # 获取邻接矩阵 adj
        adj = self.graph_embedding.gc1.adj  # 从 graph_embedding 中获取 adj

        # 调用 GCNWithMultiHeadGATAndTCN 进行图卷积、图注意力和 TCN 操作
        x = self.graph_embedding(x, adj, epoch=epoch, max_epochs=max_epochs)

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
