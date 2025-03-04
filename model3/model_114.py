import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math, os
import numpy as np

"""
Baseline+GCN+GAT
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
        # output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class MultiHeadGraphAttentionLayer(nn.Module):
    """
    多头图注意力层 (Multi-Head GAT Layer)
    """

    def __init__(self, in_features, out_features, num_heads=4, alpha=0.2):
        super(MultiHeadGraphAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads  # 每个头的输出特征维度
        self.alpha = alpha
        # 为每个头定义独立的线性变换
        self.W = nn.ModuleList([nn.Linear(in_features, self.out_per_head, bias=False) for _ in range(num_heads)])
        self.a = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.out_per_head * 2)) for _ in range(num_heads)])  # Attention weights

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=2)  # Softmax is computed over the neighbors

    def forward(self, h):
        # print(f"MultiHeadGraphAttentionLayer Forward Called: Epoch {epoch}")  # 添加调试信息
        B, N, F = h.size()
        outputs = []

        for i in range(self.num_heads):
            # 对每个头进行独立的线性变换
            h_prime = self.W[i](h)  # [B, N, F_per_head]

            # 计算注意力分数
            e = torch.matmul(h_prime, h_prime.transpose(1, 2))  # [B, N, N]
            e = self.leakyrelu(e)  # [B, N, N]
            # 不能使用这个进行约束 没法训练 损失率输出为nan
            # # 使用邻接矩阵约束注意力分数
            # e = e.masked_fill(adj == 0, float('-inf'))  # 将不存在的边的权重设为负无穷
            attention = self.softmax(e)  # Softmax on each row [B, N, N]

            # Apply attention mechanism
            h_prime = h_prime.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, F_per_head]
            h_prime = h_prime * attention.unsqueeze(-1)  # [B, N, N, F_per_head]

            # 聚合邻居信息
            output = torch.sum(h_prime, dim=2)  # [B, N, F_per_head]
            outputs.append(output)

        # 将多个头的输出拼接
        output = torch.cat(outputs, dim=-1)  # [B, N, F]

        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_heads=4):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
        # 加上GAT
        self.gat1 = MultiHeadGraphAttentionLayer(nhid, nout, num_heads)
        self.bn1 = nn.BatchNorm1d(nhid)
        # self.gc2 = GraphConvolution(nhid, nout, mat_path)
        # self.bn2 = nn.BatchNorm1d(nout)
        # self.dropout = dropout

    def forward(self, x):
        x = self.gc1(x)
        x = x.transpose(1, 2).contiguous()
        x = self.bn1(x).transpose(1, 2).contiguous()
        x = F.relu(x)

        # 加上GAT
        x = self.gat1(x)
        return x


class AUwGCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        mat_dir = '/kaggle/working/ME-GCN-Project'
        self.mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))
        adj_mat = np.load(self.mat_path)
        self.register_buffer('adj', torch.from_numpy(adj_mat))
        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, self.mat_path, num_heads=4))
        # self.graph_embedding = torch.nn.Sequential(GCN(2, 32, 32, mat_path))
        in_dim = 192  # 24

        self._sequential = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0,
                            bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            # # receptive filed: 7
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2,
                            bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )
        # 0:micro(start,end,None),    3:macro(start,end,None),
        # 6:micro_apex,7:macro_apex,  8:micro_action, macro_action
        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # (b*t, n, c)
        # TCN+GAT
        # TypeError: Sequential.forward() takes 2 positional arguments but 3 were given
        x = self.graph_embedding(x).reshape(b, t, -1).transpose(1, 2)  # (b, C=384=12*32, t)
        # x = self.graph_embedding(x).reshape(b, t, n, 16)
        x = self._sequential(x)
        x = self._classification(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)



