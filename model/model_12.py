import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
权重初始化：
使用xavier_uniform_初始化权重矩阵W和a。
GCN层添加Dropout：
在GCN中间层添加Dropout以抑制过拟合。
多头注意力层的Dropout概率：
将Dropout概率降低到0.05，减少对特征稀释。
取消输入输出特征维度限制：
多头注意力中取消强制in_features == out_features约束。
layer_num = 1
"""


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
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
        self.register_buffer('adj', torch.from_numpy(adj_mat))  # 保持与权重相同的设备

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, c = input.shape
        assert c == self.in_features, "Input feature dimension must match in_features"

        support = torch.bmm(input, self.weight.unsqueeze(0).expand(b, -1, -1))
        output = torch.bmm(self.adj.unsqueeze(0).expand(b, -1, -1), support)

        if self.bias is not None:
            output += self.bias

        return output

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        # 使用Xavier初始化权重
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # [B, N, F] x [F, F'] -> [B, N, F']
        Wh_repeat_1 = Wh.unsqueeze(2).repeat(1, 1, Wh.size(1), 1)  # [B, N, 1, F'] -> [B, N, N, F']
        Wh_repeat_2 = Wh.unsqueeze(1).repeat(1, Wh.size(1), 1, 1)  # [B, 1, N, F'] -> [B, N, N, F']
        a_input = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=-1)  # [B, N, N, 2F']

        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # [B, N, N, 2F'] x [2F', 1] -> [B, N, N]
        attention = F.softmax(e, dim=-1)  # [B, N, N] 归一化注意力权重
        h_prime = torch.bmm(attention, Wh)  # [B, N, N] x [B, N, F'] -> [B, N, F']
        return h_prime


class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.05):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        # 每个注意力头使用独立的 GraphAttentionLayer
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features) for _ in range(heads)
        ])

        # 融合多头输出的权重矩阵
        self.W = nn.Parameter(torch.Tensor(in_features * heads, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, h, adj):
        head_outputs = [head(h, adj) for head in self.attention_heads]  # 各头的输出 [B, N, F']
        h_prime = torch.cat(head_outputs, dim=-1)  # [B, N, F'*heads]
        h_prime = self.dropout(h_prime)  # 加入Dropout抑制过拟合
        h_prime = torch.matmul(h_prime, self.W)  # [B, N, F'*heads] x [F'*heads, F] -> [B, N, F]
        return h_prime


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_layers=1):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gc_layers = nn.ModuleList()

        # 构建GCN层
        for i in range(num_layers):
            in_features = nfeat if i == 0 else nhid
            out_features = nhid if i < num_layers - 1 else nout
            self.gc_layers.append(GraphConvolution(in_features, out_features, mat_path))

        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(nhid) for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(dropout)  # 在GCN层加入Dropout

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.gc_layers[i](x)
            if i < self.num_layers - 1:  # 中间层加BN和ReLU
                x = x.transpose(1, 2).contiguous()
                x = self.bn_layers[i](x).transpose(1, 2).contiguous()
                x = F.relu(x)
                x = self.dropout(x)  # 加入Dropout
        return x, self.gc_layers[-1].adj


class AUwGCN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        mat_dir = '/kaggle/working/ME-GCN-Project'
        mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # 图嵌入模块
        self.graph_embedding = GCN(2, 32, 32, mat_path, dropout=0.1, num_layers=1)

        # 多头注意力层
        self.attention = MultiHeadGraphAttentionLayer(in_features=32, out_features=32, heads=4, dropout=0.05)

        # 特征提取与分类模块
        self._sequential = nn.Sequential(
            nn.Conv1d(384, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self._classification = nn.Conv1d(128, 10, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape
        x = x.reshape(b * t, n, c)
        x, adj = self.graph_embedding(x)
        x = self.attention(x, adj)

        x = x.reshape(b, t, -1).transpose(1, 2)
        x = self._sequential(x)
        x = self._classification(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
