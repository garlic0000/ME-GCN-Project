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
            in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) * dilation // 2, dilation=dilation
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

        # 微表情分支
        self.micro_branch = nn.Sequential(
            GraphConvolution(nfeat, nhid, mat_path),
            nn.ReLU(inplace=True),
            MultiHeadGraphAttentionLayer(nhid, nout, num_heads=4, dropout=dropout),
            TCNBlock(nout, nout, kernel_size=3, dilation=2, dropout=0.2),
        )

        # 宏表情分支
        self.macro_branch = nn.Sequential(
            GraphConvolution(nfeat, nhid, mat_path),
            nn.ReLU(inplace=True),
            MultiHeadGraphAttentionLayer(nhid, nout, num_heads=4, dropout=dropout),
            TCNBlock(nout, nout, kernel_size=5, dilation=1, dropout=0.2),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(nout * 2, nout, kernel_size=1),
            nn.BatchNorm1d(nout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, adj):
        micro_feat = self.micro_branch(x)
        macro_feat = self.macro_branch(x)
        combined_feat = torch.cat([micro_feat, macro_feat], dim=1)  # 融合微表情和宏表情
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
        adj = self.graph_embedding.micro_branch[0].adj  # 共享邻接矩阵

        x = self.graph_embedding(x, adj)
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
