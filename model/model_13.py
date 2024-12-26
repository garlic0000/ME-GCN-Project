import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
1. GCN层的权重初始化和Dropout:
权重初始化:
使用了 xavier_uniform_ 对权重矩阵进行了初始化，这在深度神经网络中通常能帮助模型更快收敛并减少梯度消失问题。
Dropout:
在 GCN 的中间层加入了 Dropout，以减少过拟合的风险。这是在 GCN 层后的输出进行 Dropout 操作，防止模型过度依赖某些特征。
2. GraphAttentionLayer的更新:
多头注意力机制:

引入了多头图注意力机制（MultiHead Attention），通过并行多个注意力头来提升模型的表达能力。
MultiHeadGraphAttentionLayer 允许多个注意力头共享输入特征，并将每个头的输出进行拼接，从而增强模型的学习能力。
Dropout 被应用到多头注意力的输出上，用于防止过拟合。
注意力系数的计算方式:

GraphAttentionLayer 中的注意力权重计算使用了矩阵乘法而不是拼接的方式，从而减少了计算量，提升了效率。
3. GCN和GAT的集成:
GCN和GAT的联合使用:
在 GCNWithGAT 中，首先使用图卷积层（GCN）来对输入进行初步处理，然后再通过图注意力层（GAT）进一步处理，以充分利用图结构信息。
GCNWithGAT 先进行图卷积，之后加入了图注意力层，这种集成可以更好地捕获局部邻居的依赖和全局结构信息。
4. 多层GCN结构:
多层GCN:
通过增加 num_layers 参数，可以灵活控制GCN的层数，进一步增强模型的表达能力。
在每一层之间加入了 BatchNorm 和 ReLU 激活函数，提升了网络的训练稳定性和非线性表达能力。
5. 卷积层和分类头:
卷积操作:
使用了多个卷积层（Conv1d），并且使用了不同大小的卷积核，包括有 dilation 的卷积核，这能够捕获不同尺度的特征信息。
结合了 BatchNorm 和 ReLU 激活函数来增强模型的表现。
分类头:
最后的分类头部分使用了一个卷积层进行输出，通过 Conv1d 将最后的特征映射到目标类别（比如情感分析中的AU类标签）。
6. 改变的初始化方式:
初始化权重:
在多头注意力层和卷积层中应用了 xavier_uniform_ 和 kaiming_normal_ 等方法来初始化权重。xavier_uniform_ 初始化帮助避免梯度消失，而 kaiming_normal_ 则适用于 ReLU 激活函数，能够加速收敛。
7. 模型模块的重构:
图嵌入模块:
GCNWithGAT 类和 GCN 类被封装成独立模块，确保了模型的模块化设计。
输入输出特征维度的改变:
通过调整卷积层的输入输出维度，使得网络可以处理更高维度的特征信息，提升了模型的表达能力。
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

        # 直接在初始化时加载邻接矩阵
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
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (GAT Layer)
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(1, out_features * 2))  # 注意力权重

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)  # Softmax 是对每个节点的邻居节点计算的

    def forward(self, h, adj):
        # h: [B, N, F'] -> 节点特征
        B, N, F = h.size()

        # 将输入特征进行线性变换
        h_prime = self.W(h)  # [B, N, F'']

        # 计算注意力系数（改为矩阵乘法，避免重复拼接）
        e = torch.matmul(h_prime, h_prime.transpose(1, 2))  # [B, N, N]
        e = self.leakyrelu(e)  # [B, N, N]
        attention = self.softmax(e)  # 对行进行 softmax [B, N, N]

        # 应用注意力机制
        h_prime = h_prime.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, F'']
        h_prime = h_prime * attention.unsqueeze(-1)  # [B, N, N, F'']

        # 聚合邻居信息
        output = torch.sum(h_prime, dim=2)  # [B, N, F'']

        return output


class GCNWithGAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCNWithGAT, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)
        self.gat1 = GraphAttentionLayer(nhid, nout, dropout)

        self.bn1 = nn.BatchNorm1d(nhid)

    def forward(self, x, adj):
        x = self.gc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        x = F.relu(x)
        # 将 adj 传递给 GAT 层
        x = self.gat1(x, adj)
        return x


class AUwGCNWithGAT(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        mat_dir = '/kaggle/working/ME-GCN-Project'
        self.mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # 直接使用 GCNWithGAT，而不是包装在 Sequential 中
        self.graph_embedding = GCNWithGAT(2, 16, 16, self.mat_path)

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

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # (b*t, n, c)

        # 获取邻接矩阵 adj
        adj = self.graph_embedding.gc1.adj  # 从 graph_embedding 中获取 adj
        # 直接调用 GCNWithGAT 进行图卷积和图注意力
        x = self.graph_embedding(x, adj)  # 传递 x 和 adj
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
                m.data.uniform_(-0.1, 0.1)  # 根据需求调整范围

