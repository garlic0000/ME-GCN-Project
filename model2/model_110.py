import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
Baseline+GCN+GAT+TCN(RW)

"""


def drop_edge(adj, drop_prob=0.4, epoch=0, max_epochs=100, min_drop_prob=0.05):
    """
    直接返回
    """
    return adj


class ResidualWeight(nn.Module):
    """残差优化模块"""

    def __init__(self, initial_alpha=0.5):
        super(ResidualWeight, self).__init__()
        # 初始化比例参数为 0.5，并约束其范围为 [0, 1]
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))  # 初始值为 0.5

    def forward(self, input, residual):
        # 在每次前向传播时，确保 alpha 的值在 [0, 1] 范围内
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        return alpha * input + (1 - alpha) * residual


class GraphConvolution(nn.Module):
    """
    Simple GCN layer with Residual Connection and ResidualWeight Optimization
    """

    def __init__(self, in_features, out_features, mat_path, bias=True, drop_prob=0.4, min_drop_prob=0.05):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Load adjacency matrix
        adj_mat = np.load(mat_path)
        self.register_buffer('adj', torch.from_numpy(adj_mat))

        # DropEdge probability
        self.drop_prob = drop_prob
        self.min_drop_prob = min_drop_prob

        # 添加一个线性层，用于匹配输入和输出维度
        if in_features != out_features:
            self.residual_layer = nn.Linear(in_features, out_features, bias=False)
        else:
            self.residual_layer = None

        # 残差优化模块
        self.residual_weight = ResidualWeight()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, epoch=0, max_epochs=100):
        # print(f"GraphConvolution Forward Called: Epoch {epoch}")  # 添加调试信息
        b, n, c = input.shape

        # Apply DropEdge to adjacency matrix with dynamic probability
        adj = drop_edge(self.adj, drop_prob=self.drop_prob, epoch=epoch, max_epochs=max_epochs,
                        min_drop_prob=self.min_drop_prob)

        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(b, 1, 1))
        output = torch.bmm(adj.unsqueeze(0).repeat(b, 1, 1), support)

        if self.bias is not None:
            output += self.bias

        # Residual connection
        if self.residual_layer is not None:
            residual = self.residual_layer(input)
        else:
            residual = input

        # 使用残差优化
        return self.residual_weight(F.relu(output), residual)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class MultiHeadGraphAttentionLayer(nn.Module):
    """
    多头图注意力层 (Multi-Head GAT Layer)
    """

    def __init__(self, in_features, out_features, num_heads=4, dropout=0.6, alpha=0.2, drop_prob=0.4,
                 min_drop_prob=0.05):
        super(MultiHeadGraphAttentionLayer, self).__init__()

        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads  # 每个头的输出特征维度
        self.dropout = dropout
        self.alpha = alpha
        self.drop_prob = drop_prob  # DropEdge probability
        self.min_drop_prob = min_drop_prob  # 最小 DropEdge 概率

        # 为每个头定义独立的线性变换
        self.W = nn.ModuleList([nn.Linear(in_features, self.out_per_head, bias=False) for _ in range(num_heads)])
        self.a = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.out_per_head * 2)) for _ in range(num_heads)])  # Attention weights

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=2)  # Softmax is computed over the neighbors
        self.residual_weight = ResidualWeight()  # 残差优化模块

    def forward(self, h, adj, epoch=0, max_epochs=100):
        # print(f"MultiHeadGraphAttentionLayer Forward Called: Epoch {epoch}")  # 添加调试信息
        B, N, F = h.size()
        outputs = []

        # Apply DropEdge to adjacency matrix with dynamic probability
        adj = drop_edge(adj, drop_prob=self.drop_prob, epoch=epoch, max_epochs=max_epochs, min_drop_prob=self.min_drop_prob)

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

        # 残差连接与优化
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
    Modified GCN with Multi-Head GAT and a single TCN
    """

    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_heads=4):
        super(GCNWithMultiHeadGATAndTCN, self).__init__()

        # 第一层 GCN
        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)

        # 第一层多头 GAT 和 TCN
        self.gat1 = MultiHeadGraphAttentionLayer(nhid, nout, num_heads, dropout)
        self.tcn1 = TCNBlock(nout, nout, kernel_size=3, dilation=1, dropout=0.2)

        # BatchNorm 层
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nout)

    def forward(self, x, adj, epoch=0, max_epochs=100):
        # print(f"GCNWithMultiHeadGATAndTCN Forward Called: Epoch {epoch}")  # 添加调试信息
        # 第一层 GCN
        x = self.gc1(x, epoch=epoch, max_epochs=max_epochs)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        x = F.relu(x)

        # 第一层多头 GAT 和 TCN
        x = self.gat1(x, adj, epoch=epoch, max_epochs=max_epochs)
        x = self.tcn1(x.transpose(1, 2)).transpose(1, 2)

        # 没有第二层 TCN，直接返回
        return x


class AUwGCNWithMultiHeadGATAndTCN(torch.nn.Module):
    """
    AU detection model with GCN, Multi-Head GAT, and one TCN layer
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

            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )

        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False
        )

        self._init_weight()

    def forward(self, x, epoch=0, max_epochs=100):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # (b*t, n, c)

        # 获取邻接矩阵 adj
        adj = self.graph_embedding.gc1.adj  # 从 graph_embedding 中获取 adj

        # 调用 GCNWithMultiHeadGATAndTCN 进行图卷积、图注意力和 TCN 操作
        x = self.graph_embedding(x, adj, epoch=epoch, max_epochs=max_epochs)

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
