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

        self.heads = heads
        self.out_features = out_features
        self.in_features = in_features

        # 线性变换
        self.W = nn.Parameter(torch.Tensor(in_features, out_features * heads))  # 权重矩阵
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))  # 注意力系数的权重
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, h, adj):
        b, n, f = h.shape  # b: batch size, n: nodes, f: features

        # 线性变换：Wh
        Wh = torch.matmul(h, self.W)  # [B, N, heads * F]
        Wh = Wh.view(b, n, self.heads, self.out_features)  # [B, N, heads, out_features]

        # 计算注意力系数
        Wh_repeat_1 = Wh.unsqueeze(3).repeat(1, 1, 1, n, 1)  # [B, N, heads, N, out_features]
        Wh_repeat_2 = Wh.unsqueeze(2).repeat(1, 1, n, 1, 1)  # [B, N, heads, N, out_features]

        # 检查 Wh_repeat_1 和 Wh_repeat_2 的形状是否一致
        assert Wh_repeat_1.shape[4] == Wh_repeat_2.shape[4], \
            f"Feature dimension mismatch: {Wh_repeat_1.shape[4]} != {Wh_repeat_2.shape[4]}"

        a_input = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=-1)  # [B, N, heads, N, 2*out_features]

        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # [B, N, heads, N]

        # Softmax to normalize attention
        attention = torch.nn.functional.softmax(e, dim=-1)  # [B, N, heads, N]

        # 计算每个节点的新表示
        h_prime = torch.matmul(attention, Wh)  # [B, N, heads, out_features]
        h_prime = h_prime.view(b, n, -1)  # [B, N, heads * out_features]

        return h_prime  # 返回的是 [B, N, heads * out_features]



class AUwGCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # 服务器测试路径
        mat_dir = '/kaggle/working/ME-GCN-Project'
        mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # Graph Convolution 层
        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, mat_path))

        in_dim = 128  # 修改 in_dim 为 128

        # Graph Attention 层
        self.attention = GraphAttentionLayer(in_features=16, out_features=16, heads=8)  # heads=8，输出128

        # 处理图卷积后输出的特征
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

        # 分类层
        self._classification = torch.nn.Conv1d(64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

        # 权重初始化
        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # 将输入形状调整为 (b*t, n, c)
        x, adj = self.graph_embedding(x)  # 获取图卷积输出和邻接矩阵
        x = self.attention(x, adj)  # 通过图注意力层

        x = x.reshape(b, t, -1).transpose(1, 2)  # 调整维度为 (b, t, -1) 然后转置
        x = self._sequential(x)  # 通过卷积层处理
        x = self._classification(x)  # 最终的分类层

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
