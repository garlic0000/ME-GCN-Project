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

多头注意力：通过引入 heads 参数，为每个头初始化一个权重矩阵 W，并将多个头的输出拼接到一起。Wh 被重塑为 [B, N, heads, F] 形式，以便进行多头处理。
LeakyReLU 激活函数：保持原有的激活函数来计算注意力分数 e，并使用 leaky_relu 进行非线性变换。
批量矩阵乘法（batch matrix multiplication）：通过对 attention 和 Wh 进行批量矩阵乘法，计算加权求和 h_prime。
拼接多个头的输出：在最后将多个头的输出拼接在一起，形成最终的节点表示。

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
    def __init__(self, in_features, out_features, heads=8, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()

        # 输入输出通道数相同，保证不改变通道数
        assert in_features == out_features, "in_features and out_features should be the same to preserve channel size"

        self.heads = heads
        self.dropout = dropout

        # 为多头注意力准备多个权重矩阵 W
        self.W = nn.Parameter(torch.Tensor(in_features, out_features * heads))  # [in_features, out_features * heads]
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))  # [2 * out_features, 1]

        # 对每个头进行注意力的参数初始化
        self.reset_parameters()
        self.output_proj = nn.Linear(out_features * heads, in_features)  # 新增：调整输出回 in_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, h, adj):
        # 输入 h 的形状: [B, N, F]
        Wh = torch.matmul(h, self.W)  # Wh: [B, N, F * heads]
        Wh = Wh.view(Wh.size(0), Wh.size(1), self.heads, -1)  # 重塑为 [B, N, heads, F]

        # 计算 pairwise attention scores
        Wh_repeat_1 = Wh.unsqueeze(2).repeat(1, 1, Wh.size(1), 1, 1)  # Shape: [B, N, N, heads, F]
        Wh_repeat_2 = Wh.unsqueeze(1).repeat(1, Wh.size(1), 1, 1, 1)  # Shape: [B, N, N, heads, F]
        a_input = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=-1)  # Shape: [B, N, N, heads, 2F]

        # 计算注意力分数
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # e: [B, N, N, heads]

        # 通过 softmax 归一化注意力分数
        attention = torch.nn.functional.softmax(e, dim=1)  # attention: [B, N, N, heads]

        # 使用注意力权重进行加权求和
        h_prime = torch.bmm(attention.view(-1, attention.size(1), attention.size(2)),
                            Wh.view(-1, Wh.size(1), Wh.size(3)))  # [B, N, F * heads]

        h_prime = h_prime.view(h_prime.size(0), h_prime.size(1), self.heads, -1)  # Reshape: [B, N, heads, F]

        # 将多个头拼接
        h_prime = h_prime.view(h_prime.size(0), h_prime.size(1), -1)  # [B, N, F * heads]
        # 新增线性投影，将输出调整为指定通道数
        h_prime = self.output_proj(h_prime)  # [B, N, in_features]

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
        # 这部分已经移到 __init__ 中
        self.conv1d = torch.nn.Conv1d(in_channels=16, out_channels=in_dim, kernel_size=1, bias=False)
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

        # 新增：适配 in_dim 的线性或卷积操作
        x = x.permute(0, 2, 1)  # [B, N, F] -> [B, F, N]，适配 Conv1d
        x = self.conv1d(x)  # 应用已定义的 Conv1d 层
        x = x.permute(0, 2, 1)  # [B, F, N] -> [B, N, F]

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
