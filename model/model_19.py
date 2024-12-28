import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
在model_18基础上引入 Non-Local Block 捕捉全局特征；
"""


class GraphConvolution(nn.Module):
    """
    Simple GCN layer with Residual Connection
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

        # Load adjacency matrix
        adj_mat = np.load(mat_path)
        self.register_buffer('adj', torch.from_numpy(adj_mat))

        # 添加一个线性层，用于匹配输入和输出维度
        if in_features != out_features:
            self.residual_layer = nn.Linear(in_features, out_features, bias=False)
        else:
            self.residual_layer = None

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
            output += self.bias

        # Residual connection
        if self.residual_layer is not None:
            residual = self.residual_layer(input)
        else:
            residual = input

        return F.relu(output + residual)  # 残差连接后激活

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


class TCNBlock(nn.Module):
    """
    TCN layer with Residual Connection
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

    def forward(self, x):
        residual = x  # 保存残差
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connection
        if self.residual_layer is not None:
            residual = self.residual_layer(residual)

        return F.relu(x + residual)  # 残差连接后激活


class NonLocalBlock(nn.Module):
    """
    Non-Local Block to capture global features
    """

    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.theta = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.g = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_conv = nn.Conv1d(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, t = x.size()  # 输入尺寸 (batch, channel, time)
        theta_x = self.theta(x).view(b, c // 2, -1)  # (b, c//2, t)
        phi_x = self.phi(x).view(b, c // 2, -1).transpose(1, 2)  # (b, t, c//2)
        g_x = self.g(x).view(b, c // 2, -1)  # (b, c//2, t)

        # Affinity matrix
        affinity = torch.bmm(phi_x, theta_x)  # (b, t, c//2) x (b, c//2, t) -> (b, t, t)
        affinity = self.softmax(affinity)  # 对列进行softmax

        # Weighted sum
        out = torch.bmm(affinity, g_x.transpose(1, 2))  # (b, t, t) x (b, t, c//2) -> (b, t, c//2)
        out = out.transpose(1, 2).view(b, c // 2, t)  # 转置回来，得到 (b, c//2, t)

        # Final output
        out = self.out_conv(out)  # (b, c, t)
        return x + out  # 残差连接


class GCNWithGATAndTCN(nn.Module):
    """
    Modified GCN with Residual Connection and a single TCN
    """

    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCNWithGATAndTCN, self).__init__()

        # 第一层 GCN
        self.gc1 = GraphConvolution(nfeat, nhid, mat_path)

        # 第一层 GAT 和 TCN
        self.gat1 = GraphAttentionLayer(nhid, nout, dropout)
        self.tcn1 = TCNBlock(nout, nout, kernel_size=3, dilation=1, dropout=0.2)

        # 去掉第二层 TCN（tcn2）

        # BatchNorm 层
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nout)

    def forward(self, x, adj):
        # 第一层 GCN
        x = self.gc1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        x = F.relu(x)

        # 第一层 GAT 和 TCN
        x = self.gat1(x, adj)
        x = self.tcn1(x.transpose(1, 2)).transpose(1, 2)

        # 没有第二层 TCN，直接返回
        return x


class AUwGCNWithGATAndTCN(torch.nn.Module):
    """
    AU detection model with GCN, GAT, one TCN layer, and Non-Local Block
    """

    def __init__(self, opt):
        super().__init__()

        mat_dir = '/kaggle/working/ME-GCN-Project'
        self.mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # 使用修改后的 GCNWithGATAndTCN
        self.graph_embedding = GCNWithGATAndTCN(2, 16, 16, self.mat_path)

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

        # 添加 Non-Local Block
        self.non_local = NonLocalBlock(64)

        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False
        )

        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.reshape(b * t, n, c)  # (b*t, n, c)

        # 获取邻接矩阵 adj
        adj = self.graph_embedding.gc1.adj  # 从 graph_embedding 中获取 adj
        # 调用 GCNWithGATAndTCN 进行图卷积、图注意力和 TCN 操作
        x = self.graph_embedding(x, adj)
        # reshape 处理为适合卷积输入的维度
        x = x.reshape(b, t, -1).transpose(1, 2)
        # 卷积操作
        x = self._sequential(x)
        # 应用 Non-Local Block
        x = self.non_local(x)
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
