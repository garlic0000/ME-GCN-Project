import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
关键更改：
在model_2的基础上
通道数不变：

将 GraphAttentionLayer 替换为 SelfAttentionLayer。
自注意力机制不再依赖邻接矩阵，而是根据特征之间的相似性进行计算。
计算的每一步保持与原模型相似的结构和行为，只是用自注意力代替了图注意力。

自注意力计算：

使用 Query (Q)、Key (K) 和 Value (V) 来计算自注意力，公式如下
这里，Q、𝐾和 𝑉通过输入特征计算得出，并且进行缩放和归一化。
无邻接矩阵：

自注意力机制直接计算输入特征之间的关系，不再依赖外部的邻接矩阵。
不再依赖图结构：

由于自注意力是基于特征之间的关系而非图结构进行的，因此不需要显式地传递邻接矩阵。
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


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SelfAttentionLayer, self).__init__()

        # 输入输出通道数相同，保证不改变通道数
        assert in_features == out_features, "in_features and out_features should be the same to preserve channel size"

        # 权重矩阵 Q, K, V
        self.W_q = nn.Parameter(torch.Tensor(in_features, out_features))  # Query
        self.W_k = nn.Parameter(torch.Tensor(in_features, out_features))  # Key
        self.W_v = nn.Parameter(torch.Tensor(in_features, out_features))  # Value
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W_q.size(1))
        self.W_q.data.uniform_(-stdv, stdv)
        self.W_k.data.uniform_(-stdv, stdv)
        self.W_v.data.uniform_(-stdv, stdv)

    def forward(self, h):
        # 输入 h 的形状: [B, N, F]

        # 计算 Q, K, V
        Q = torch.matmul(h, self.W_q)  # Shape: [B, N, F]
        K = torch.matmul(h, self.W_k)  # Shape: [B, N, F]
        V = torch.matmul(h, self.W_v)  # Shape: [B, N, F]

        # 计算 Q 和 K 的点积注意力得分
        attention_scores = torch.matmul(Q, K.transpose(1, 2))  # Shape: [B, N, N]

        # 缩放
        attention_scores = attention_scores / math.sqrt(K.size(-1))

        # 使用 softmax 计算归一化的注意力分数
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)  # Shape: [B, N, N]

        # 使用注意力分数加权求和 V
        h_prime = torch.matmul(attention_weights, V)  # Shape: [B, N, F]

        return h_prime


class AUwGCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # 服务器测试路径
        mat_dir = '/kaggle/working/ME-GCN-Project'
        mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, mat_path))

        in_dim = 192  # 保持输入通道数为192

        self.attention = SelfAttentionLayer(in_features=16, out_features=16)  # 使用自注意力层

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
        x, adj = self.graph_embedding(x)  # 获取图卷积的输出
        x = self.attention(x)  # 传入自注意力层进行处理

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



