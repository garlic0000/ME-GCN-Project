import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np
from torch.nn.utils import weight_norm

"""
主要改动和优化说明：
Graph Convolution:

使用 广播 来避免重复计算。
添加了 残差连接（Residual Connection），避免信息丢失并促进梯度流。
将 repeat 改为 expand，节省显存。
GCN模块:

在 GCN 类中重新启用 Dropout，并确保其在训练时有效。
调整了数据格式的处理，以便 BatchNorm1d 使用。
权重初始化:

使用 torch.nn.init.kaiming_normal_ 初始化卷积层的权重，增加训练稳定性。
在 GCN 层上应用了 权重归一化（weight_norm），进一步提升模型的稳定性。
功能改进:

在 GraphConvolution 层添加 残差连接，有助于缓解梯度消失问题。
在 AUwGCN 中的卷积层添加了适当的 批归一化（BatchNorm1d）来改善训练过程中的收敛性。

"""

class GraphConvolution(nn.Module):
    """
    Simple GCN layer with optimizations:
    - Using broadcast to avoid repeat()
    - Adding residual connection to mitigate vanishing gradients
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

        adj_mat = np.load(mat_path)
        self.register_buffer('adj', torch.from_numpy(adj_mat))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, c = input.shape
        support = torch.bmm(input, self.weight.unsqueeze(0).expand(b, -1, -1))
        output = torch.bmm(self.adj.unsqueeze(0).expand(b, -1, -1), support)

        if self.bias is not None:
            output = output + self.bias

        # Adding residual connection (skip connection)
        return F.relu(output + input)  # Residual connection

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3):
        super(GCN, self).__init__()

        self.gc1 = weight_norm(GraphConvolution(nfeat, nhid, mat_path))  # Applying weight normalization
        self.bn1 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, x):
        x = self.gc1(x)
        x = x.transpose(1, 2).contiguous()  # BatchNorm1d expects the input in (B, C, T) format
        x = self.bn1(x).transpose(1, 2).contiguous()
        x = F.relu(x)

        # Apply dropout during training to regularize the model
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class AUwGCN(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # File path to adjacency matrix (graph structure)
        mat_dir = '/kaggle/working/ME-GCN-Project'
        mat_path = os.path.join(mat_dir, 'assets', f'{opt["dataset"]}.npy')

        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, mat_path))

        in_dim = 192  # Features after GCN processing

        self._sequential = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            # Receptive field: 7
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

        # Reshape to (b*t, n, c) to pass through GCN
        x = x.reshape(b * t, n, c)
        x = self.graph_embedding(x).reshape(b, t, -1).transpose(1, 2)

        # Sequential layers (Conv1d, BatchNorm, ReLU)
        x = self._sequential(x)

        # Classification output layer
        x = self._classification(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

