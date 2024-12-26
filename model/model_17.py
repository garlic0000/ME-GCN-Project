import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np


# Graph Convolution Layer
class GraphConvolution(nn.Module):
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

        # 可学习的邻接矩阵
        adj_mat = np.load(mat_path)
        self.adj = Parameter(torch.from_numpy(adj_mat).float(), requires_grad=True)

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
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    单头的图注意力层 (GAT Layer)，基于 Velickovic 等人的论文: "Graph Attention Networks"
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        """
        Args:
            in_features (int): 输入特征的维度
            out_features (int): 输出特征的维度
            dropout (float): Dropout 概率
            alpha (float): LeakyReLU 中的负斜率
            concat (bool): 是否在激活后进行拼接
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 权重矩阵 W 和 注意力权重 a
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        # LeakyReLU 激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # 参数初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """初始化权重参数"""
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        """
        前向传播
        Args:
            h (torch.Tensor): 输入特征矩阵，形状为 [N, in_features]
            adj (torch.Tensor): 邻接矩阵，形状为 [N, N]

        Returns:
            h_prime (torch.Tensor): 输出特征矩阵，形状为 [N, out_features]
        """
        Wh = torch.bmm(h,
                       self.W.unsqueeze(0).repeat(h.size(0), 1, 1))  # 线性变换: [B, N, in_features] -> [B, N, out_features]
        N = Wh.size(1)

        # 计算注意力系数 e_ij
        a_input = torch.cat([Wh.repeat(1, N, 1), Wh.repeat(N, 1, 1)], dim=2)  # [B, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [B, N]

        # 仅保留邻接矩阵中有连接的注意力权重
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 聚合邻接节点特征
        h_prime = torch.bmm(attention.unsqueeze(1), Wh)  # [B, 1, N] * [B, N, out_features] = [B, 1, out_features]

        if self.concat:
            return F.elu(h_prime.squeeze(1))  # 激活并返回输出特征
        else:
            return h_prime.squeeze(1)  # 不进行拼接的情况下，直接返回

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# Multi-Head Graph Attention Layer
class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.6, alpha=0.2):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha) for _ in range(num_heads)
        ])
        self.fc = nn.Linear(out_features * num_heads, out_features)

    def forward(self, h, adj):
        h_heads = [head(h, adj) for head in self.attention_heads]
        h_concat = torch.cat(h_heads, dim=-1)  # Concatenate outputs from all heads
        h_out = self.fc(h_concat)  # Linear layer to aggregate heads
        return h_out


# Temporal Convolutional Block (TCN) with Residual Connection
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + residual


# Dual-Branch GCN with GAT and TCN Integration
class DualBranchGCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(DualBranchGCN, self).__init__()

        # 加载邻接矩阵
        adj_mat = np.load(mat_path)
        self.adj = nn.Parameter(torch.from_numpy(adj_mat).float(), requires_grad=True)

        # 微表情分支
        self.micro_gcn = GraphConvolution(nfeat, nhid, mat_path)
        self.micro_attention = MultiHeadGraphAttentionLayer(nhid, nout, num_heads=4, dropout=dropout)
        self.micro_tcn = TCNBlock(nout, nout, kernel_size=3, dilation=2, dropout=0.2)

        # 宏表情分支
        self.macro_gcn = GraphConvolution(nfeat, nhid, mat_path)
        self.macro_attention = MultiHeadGraphAttentionLayer(nhid, nout, num_heads=4, dropout=dropout)
        self.macro_tcn = TCNBlock(nout, nout, kernel_size=5, dilation=1, dropout=0.2)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(nout * 2, nout, kernel_size=1),
            nn.BatchNorm1d(nout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 获取邻接矩阵
        adj = self.adj

        # 微表情分支
        micro_feat = self.micro_gcn(x)
        micro_feat = F.relu(micro_feat)
        micro_feat = self.micro_attention(micro_feat, adj)
        micro_feat = self.micro_tcn(micro_feat)

        # 宏表情分支
        macro_feat = self.macro_gcn(x)
        macro_feat = F.relu(macro_feat)
        macro_feat = self.macro_attention(macro_feat, adj)
        macro_feat = self.macro_tcn(macro_feat)

        # 融合微表情和宏表情
        combined_feat = torch.cat([micro_feat, macro_feat], dim=1)
        return self.fusion(combined_feat)


# Final Model
class AUwGCNWithGATAndTCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        mat_dir = '/kaggle/working/ME-GCN-Project'
        self.mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        self.graph_embedding = DualBranchGCN(2, 16, 32, self.mat_path)

        in_dim = 192
        self._sequential = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )

        # Temporal attention module
        self.attention = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.Softmax(dim=2)
        )

        # Final classification layer
        self._classification = torch.nn.Conv1d(
            64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False
        )

        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape
        x = x.reshape(b * t, n, c)
        adj = self.graph_embedding.micro_gcn.adj  # 通过 micro_gcn 获取邻接矩阵

        x = self.graph_embedding(x)
        x = x.reshape(b, t, -1).transpose(1, 2)
        x = self._sequential(x)

        # Temporal attention weighting
        attention_weights = self.attention(x)
        x = x * attention_weights

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
