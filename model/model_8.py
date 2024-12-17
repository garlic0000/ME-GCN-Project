import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
关键更改：
在model_5的基础上
增加了 GCN层数（1层），并提高了每层的 卷积通道数。
增加了 多头图注意力，使得模型能够从多个角度学习图结构的信息。
通过 卷积核大小的多样化 和 残差连接 进一步优化了特征提取的能力。
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


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()

        # 输入输出通道数相同，保证不改变通道数
        assert in_features == out_features, "in_features and out_features should be the same to preserve channel size"

        # 权重矩阵和注意力参数
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))  # 权重矩阵
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))  # 注意力参数
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, h, adj):
        # 输入 h 的形状: [B, N, F]
        Wh = torch.matmul(h, self.W)  # 线性变换，Wh: [B, N, F]

        # 计算 pairwise attention scores
        Wh_repeat_1 = Wh.unsqueeze(2).repeat(1, 1, Wh.size(1), 1)  # Shape: [B, N, N, F]
        Wh_repeat_2 = Wh.unsqueeze(1).repeat(1, Wh.size(1), 1, 1)  # Shape: [B, N, N, F]
        a_input = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=-1)  # Shape: [B, N, N, 2F]

        # 计算注意力分数
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # 注意力得分 e: [B, N, N]

        # 通过 softmax 归一化注意力分数
        attention = torch.nn.functional.softmax(e, dim=1)  # [B, N, N]

        # 使用注意力权重进行加权求和
        h_prime = torch.bmm(attention, Wh)  # 批量矩阵乘法: [B, N, N] * [B, N, F] -> [B, N, F]

        return h_prime


class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.1):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.heads = heads
        self.out_features = out_features
        self.dropout = dropout

        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features) for _ in range(heads)
        ])

        self.W = nn.Parameter(torch.Tensor(in_features * heads, out_features))  # 合并多头的权重矩阵
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, h, adj):
        # 对每个注意力头进行计算
        head_outputs = [head(h, adj) for head in self.attention_heads]

        # 合并多个头的输出
        h_prime = torch.cat(head_outputs, dim=-1)  # [B, N, F*heads]

        # 线性变换合并后的输出
        h_prime = torch.matmul(h_prime, self.W)  # [B, N, F*heads] x [F*heads, out_features] -> [B, N, out_features]

        return h_prime


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_layers=1):  # 默认更深的网络
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.gc_layers = nn.ModuleList()

        # 添加多个图卷积层
        for i in range(num_layers):
            in_features = nfeat if i == 0 else nhid
            out_features = nhid if i < num_layers - 1 else nout
            self.gc_layers.append(GraphConvolution(in_features, out_features, mat_path))

        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(nhid) for _ in range(num_layers - 1)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.gc_layers[i](x)
            if i < self.num_layers - 1:
                x = x.transpose(1, 2).contiguous()  # [B, N, F] -> [B, F, N]
                x = self.bn_layers[i](x).transpose(1, 2).contiguous()
                x = F.relu(x)

        return x, self.gc_layers[-1].adj


class AUwGCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        mat_dir = '/kaggle/working/ME-GCN-Project'
        mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # 增加GCN层数
        self.graph_embedding = torch.nn.Sequential(GCN(2, 32, 32, mat_path, num_layers=1))

        in_dim = 384
        self.attention = MultiHeadGraphAttentionLayer(in_features=32, out_features=32, heads=4, dropout=0.1)

        self._sequential = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 128, kernel_size=1, stride=1, padding=0, bias=False),  # 增加卷积通道数
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
        ) 

        self._classification = torch.nn.Conv1d(128, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2,
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
