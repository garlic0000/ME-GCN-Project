import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
去掉ResidualWeight
但有普通的残差连接
drop_prob=0.4
min_drop_prob=0.05
"""

def drop_edge(adj, drop_prob=0.4, epoch=0, max_epochs=100, min_drop_prob=0.05):
    """动态调整 DropEdge 概率"""
    dynamic_prob = max(drop_prob * (1 - epoch / max_epochs), min_drop_prob)
    mask = torch.rand_like(adj, dtype=torch.float32) > dynamic_prob
    return adj * mask


class GraphConvolution(nn.Module):
    """GCN 层，使用普通残差连接"""

    def __init__(self, in_features, out_features, mat_path, bias=True, drop_prob=0.4, min_drop_prob=0.05):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features)) if bias else None
        self.reset_parameters()

        adj_mat = np.load(mat_path)
        self.register_buffer('adj', torch.from_numpy(adj_mat))
        self.drop_prob = drop_prob
        self.min_drop_prob = min_drop_prob

        self.residual_layer = nn.Linear(in_features, out_features, bias=False) if in_features != out_features else None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, epoch=0, max_epochs=100):
        b, n, c = input.shape
        adj = drop_edge(self.adj, self.drop_prob, epoch, max_epochs, self.min_drop_prob)
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        output = torch.bmm(adj.unsqueeze(0).repeat(b, 1, 1), support)
        if self.bias is not None:
            output += self.bias
        residual = self.residual_layer(input) if self.residual_layer else input
        return F.relu(output + residual)


class MultiHeadGraphAttentionLayer(nn.Module):
    """多头图注意力层，使用普通残差连接"""

    def __init__(self, in_features, out_features, num_heads=4, dropout=0.6, alpha=0.2, drop_prob=0.4,
                 min_drop_prob=0.05):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.drop_prob = drop_prob
        self.min_drop_prob = min_drop_prob
        self.W = nn.ModuleList([nn.Linear(in_features, self.out_per_head, bias=False) for _ in range(num_heads)])
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, h, adj, epoch=0, max_epochs=100):
        B, N, F = h.size()
        adj = drop_edge(adj, self.drop_prob, epoch, max_epochs, self.min_drop_prob)
        outputs = [torch.matmul(self.softmax(self.leakyrelu(torch.matmul(self.W[i](h), self.W[i](h).transpose(1, 2)))),
                                self.W[i](h)) for i in range(self.num_heads)]
        return torch.cat(outputs, dim=-1) + h  # 残差连接


class TCNBlock(nn.Module):
    """TCN 层，使用普通残差连接"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1,
                              padding=(kernel_size - 1) * dilation // 2, dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                        bias=False) if in_channels != out_channels else None

    def forward(self, x):
        residual = self.residual_layer(x) if self.residual_layer else x
        return self.relu(self.dropout(self.batch_norm(self.conv(x))) + residual)


class GCNWithMultiHeadGATAndTCN(nn.Module):
    """结合 GCN、GAT 和 TCN"""

    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_heads=4):
        super(GCNWithMultiHeadGATAndTCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
        self.gat1 = MultiHeadGraphAttentionLayer(nhid, nout, num_heads, dropout)
        self.tcn1 = TCNBlock(nout, nout, kernel_size=3, dilation=1, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(nhid)

    def forward(self, x, adj, epoch=0, max_epochs=100):
        x = self.bn1(self.gc1(x, epoch, max_epochs).transpose(1, 2)).transpose(1, 2)
        x = self.tcn1(self.gat1(x, adj, epoch, max_epochs).transpose(1, 2)).transpose(1, 2)
        return x


class AUwGCNWithMultiHeadGATAndTCN(torch.nn.Module):
    """AU 检测模型"""

    def __init__(self, opt):
        super().__init__()
        mat_dir = '/kaggle/working/ME-GCN-Project'
        self.mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))
        self.graph_embedding = GCNWithMultiHeadGATAndTCN(2, 16, 16, self.mat_path, num_heads=4)
        in_dim = 192
        self._sequential = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=1, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2, bias=False), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2, bias=False), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self._classification = nn.Conv1d(64, 10, kernel_size=3, padding=2, dilation=2, bias=False)

    def forward(self, x, epoch=0, max_epochs=100):
        b, t, n, c = x.shape
        x = x.reshape(b * t, n, c)
        x = self.graph_embedding(x, self.graph_embedding.gc1.adj, epoch, max_epochs)
        x = self._sequential(x.reshape(b, t, -1).transpose(1, 2))
        return self._classification(x)
