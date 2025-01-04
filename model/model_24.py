import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
在model_23基础上

使用 GraphSAGE 和 GATv2


"""


def drop_edge(adj, drop_prob=0.1, epoch=0, max_epochs=100):
    """动态调整 DropEdge 概率"""
    dynamic_prob = drop_prob * (1 - epoch / max_epochs)
    mask = torch.rand_like(adj, dtype=torch.float32) > dynamic_prob
    return adj * mask


class ResidualWeight(nn.Module):
    """残差优化模块"""

    def __init__(self):
        super(ResidualWeight, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化比例参数为 0.5

    def forward(self, input, residual):
        return self.alpha * input + (1 - self.alpha) * residual


class GraphSAGEConv(nn.Module):
    def __init__(self, in_features, out_features, aggregation='mean', drop_prob=0.1):
        super(GraphSAGEConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregation = aggregation
        self.drop_prob = drop_prob
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def aggregate(self, neighbors):
        if self.aggregation == 'mean':
            return torch.mean(neighbors, dim=1)
        elif self.aggregation == 'sum':
            return torch.sum(neighbors, dim=1)
        elif self.aggregation == 'max':
            return torch.max(neighbors, dim=1)[0]
        else:
            raise ValueError("Unknown aggregation method")

    def forward(self, input, adj):
        b, n, c = input.shape

        # Apply DropEdge to adjacency matrix
        adj = drop_edge(adj, drop_prob=self.drop_prob)

        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))  # [B, N, F_out]
        output = torch.bmm(adj.unsqueeze(0).repeat(b, 1, 1), support)  # [B, N, F_out]

        if self.bias is not None:
            output += self.bias

        return F.relu(output)



class GATv2Layer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.6, alpha=0.2, drop_prob=0.1):
        super(GATv2Layer, self).__init__()

        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.drop_prob = drop_prob

        self.W = nn.ModuleList([nn.Linear(in_features, self.out_per_head, bias=False) for _ in range(num_heads)])
        self.a = nn.ParameterList([nn.Parameter(torch.zeros(1, self.out_per_head * 2)) for _ in range(num_heads)])

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=2)
        self.residual_weight = ResidualWeight()

    def forward(self, h, adj):
        B, N, F = h.size()
        outputs = []

        adj = drop_edge(adj, drop_prob=self.drop_prob)  # Apply drop_edge

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

        return self.residual_weight(output, h)


class TCNBlock(nn.Module):
    """
    TCN layer with ResidualWeight Optimization
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

        # 添加一个卷积层，用于匹配输入和输出维度
        if in_channels != out_channels:
            self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_layer = None

        # 残差优化模块
        self.residual_weight = ResidualWeight()

    def forward(self, x):
        residual = x  # 保存残差
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connection with optimization
        if self.residual_layer is not None:
            residual = self.residual_layer(residual)

        return self.residual_weight(x, residual)


class GCNWithMultiHeadGATAndTCN(nn.Module):
    """
    Modified GCN with GraphSAGE and GATv2 with a single TCN layer
    """

    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_heads=4):
        super(GCNWithMultiHeadGATAndTCN, self).__init__()

        # 第一层 GraphSAGE
        self.gc1 = GraphSAGEConv(nfeat, nhid, aggregation='mean', drop_prob=dropout)

        # 第一层 GATv2 和 TCN
        self.gat1 = GATv2Layer(nhid, nout, num_heads=num_heads, dropout=dropout)
        self.tcn1 = TCNBlock(nout, nout, kernel_size=3, dilation=1, dropout=0.2)

        # BatchNorm 层
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nout)

    def forward(self, x, adj):
        # 第一层 GraphSAGE
        x = self.gc1(x, adj)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        x = F.relu(x)

        # 第一层 GATv2 和 TCN
        x = self.gat1(x, adj)
        x = self.tcn1(x.transpose(1, 2)).transpose(1, 2)

        return x


class AUwGCNWithMultiHeadGATAndTCN(torch.nn.Module):
    """
    AU detection model with GraphSAGE, GATv2, and one TCN layer
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
        )

        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False
        )

        self._init_weight()

    def forward(self, x, adj):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # (b*t, n, c)

        # 将邻接矩阵传递给 graph_embedding 的 forward 函数
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


