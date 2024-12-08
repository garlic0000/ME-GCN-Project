import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
修正了注意力归一化方向。
添加了数值稳定性优化（e - max）。
初始化调整：权重初始化改为 xavier_uniform_。
加入 Dropout：支持图卷积网络中的 Dropout。
移除不必要的 Sequential：graph_embedding 直接调用而不是用 torch.nn.Sequential 包装。
"""

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
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
        self.register_buffer('adj', torch.from_numpy(adj_mat))  # 保持与权重相同的设备

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, c = input.shape
        assert c == self.in_features, "Input feature dimension must match in_features"

        support = torch.bmm(input, self.weight.unsqueeze(0).expand(b, -1, -1))
        output = torch.bmm(self.adj.unsqueeze(0).expand(b, -1, -1), support)

        if self.bias is not None:
            output += self.bias

        return output

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} -> {self.out_features})'


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        assert in_features == out_features, "Input and output feature dimensions must match"
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # Linear transformation
        Wh_repeat_1 = Wh.unsqueeze(2).repeat(1, 1, Wh.size(1), 1)
        Wh_repeat_2 = Wh.unsqueeze(1).repeat(1, Wh.size(1), 1, 1)
        a_input = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=-1)

        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))
        e = e - torch.max(e, dim=-1, keepdim=True).values  # 数值稳定性优化
        attention = torch.nn.functional.softmax(e, dim=-1)  # 注意归一化方向

        h_prime = torch.bmm(attention, Wh)
        return h_prime


class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.1):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.heads = heads
        self.out_features = out_features
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features) for _ in range(heads)
        ])
        self.W = nn.Parameter(torch.Tensor(in_features * heads, out_features))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, h, adj):
        head_outputs = [head(h, adj) for head in self.attention_heads]
        h_prime = torch.cat(head_outputs, dim=-1)
        h_prime = self.dropout(torch.matmul(h_prime, self.W))
        return h_prime


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_layers=1):
        super(GCN, self).__init__()
        assert num_layers > 0, "Number of layers must be at least 1"

        self.gc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            in_features = nfeat if i == 0 else nhid
            out_features = nhid if i < num_layers - 1 else nout
            self.gc_layers.append(GraphConvolution(in_features, out_features, mat_path))
            if i < num_layers - 1:
                self.bn_layers.append(nn.BatchNorm1d(out_features))

    def forward(self, x):
        for i, gc_layer in enumerate(self.gc_layers):
            x = gc_layer(x)
            if i < len(self.gc_layers) - 1:
                x = x.transpose(1, 2).contiguous()
                x = self.bn_layers[i](x).transpose(1, 2).contiguous()
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x, self.gc_layers[-1].adj


class AUwGCN(nn.Module):
    def __init__(self, opt):
        super(AUwGCN, self).__init__()
        mat_dir = '/kaggle/working/ME-GCN-Project'
        mat_path = os.path.join(mat_dir, 'assets', f"{opt['dataset']}.npy")

        self.graph_embedding = GCN(2, 32, 32, mat_path, dropout=0.3, num_layers=2)
        self.attention = MultiHeadGraphAttentionLayer(in_features=32, out_features=32, heads=4, dropout=0.1)

        in_dim = 384
        self._sequential = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self._classification = nn.Conv1d(128, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self._init_weight()

    def forward(self, x):
        b, t, n, c = x.shape
        x = x.view(b * t, n, c)
        x, adj = self.graph_embedding(x)
        x = self.attention(x, adj)
        x = x.view(b, t, -1).transpose(1, 2)
        x = self._sequential(x)
        x = self._classification(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
