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
    def __init__(self, in_features, out_features, heads=8):
        super(GraphAttentionLayer, self).__init__()

        self.heads = heads  # 设置多头注意力
        self.out_features = out_features
        self.in_features = in_features

        self.W = nn.Parameter(torch.Tensor(in_features, out_features * heads))  # 多头输出特征数
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))  # 注意力系数
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, h, adj):
        b, n, f = h.shape
        print(f"Input h shape: {h.shape}")  # 打印输入形状

        # 线性变换
        Wh = torch.matmul(h, self.W)  # [B, N, heads * F]
        print(f"Wh shape after linear transformation: {Wh.shape}")  # 打印 Wh 的形状

        # 确保特征数与头数兼容
        f = self.out_features  # 每个头的特征数量
        Wh = Wh.view(b, n, self.heads, f)  # 重新调整为 [B, N, heads, F]
        print(f"Wh shape after view: {Wh.shape}")  # 打印调整后的形状

        # 计算注意力分数
        Wh_repeat_1 = Wh.unsqueeze(3).repeat(1, 1, 1, n, 1)  # Shape: [B, N, heads, N, F]
        Wh_repeat_2 = Wh.unsqueeze(2).repeat(1, 1, n, 1, 1)  # Shape: [B, N, heads, N, F]

        print(f"Wh_repeat_1 shape: {Wh_repeat_1.shape}")  # 打印 Wh_repeat_1 的形状
        print(f"Wh_repeat_2 shape: {Wh_repeat_2.shape}")  # 打印 Wh_repeat_2 的形状

        # 确保拼接前维度匹配
        # 调整维度，确保拼接时匹配
        Wh_repeat_1 = Wh_repeat_1.permute(0, 1, 3, 2, 4)  # [B, N, N, heads, F]
        Wh_repeat_2 = Wh_repeat_2.permute(0, 1, 3, 2, 4)  # [B, N, N, heads, F]

        print(f"Wh_repeat_1 permuted shape: {Wh_repeat_1.shape}")
        print(f"Wh_repeat_2 permuted shape: {Wh_repeat_2.shape}")

        # 确保拼接时，所有维度都匹配
        a_input = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=-1)  # Shape: [B, N, N, heads, 2F]
        print(f"a_input shape after concat: {a_input.shape}")  # 打印拼接后的形状

        # 计算注意力分数
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # e: [B, N, heads, N]

        # 通过 softmax 归一化注意力分数
        attention = torch.nn.functional.softmax(e, dim=-1)  # [B, N, heads, N]

        # 使用注意力权重进行加权求和
        h_prime = torch.matmul(attention, Wh)  # 批量矩阵乘法: [B, N, heads, N] * [B, N, heads, F] -> [B, N, heads, F]

        # 拼接多个头的输出
        h_prime = h_prime.view(b, n, -1)  # 拼接多个头的输出: [B, N, heads * F]

        return h_prime






class AUwGCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # 服务器测试路径
        mat_dir = '/kaggle/working/ME-GCN-Project'
        mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, mat_path))

        in_dim = 192  # 保持输入通道数为192

        self.attention = GraphAttentionLayer(in_features=16, out_features=16, heads=8)  # 调整 GraphAttentionLayer

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

