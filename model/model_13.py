import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
增加GCN层数并加入残差连接。
优化多头注意力层，拼接原始输入和多头输出。
分类头改为全局池化+线性层，简化架构。
使用kaiming初始化权重，提升训练稳定性。
Dropout增加到注意力层和特征提取模块中，防止过拟合。
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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

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
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # [B, N, F] x [F, F'] -> [B, N, F']
        Wh_repeat_1 = Wh.unsqueeze(2).repeat(1, 1, Wh.size(1), 1)  # [B, N, 1, F'] -> [B, N, N, F']
        Wh_repeat_2 = Wh.unsqueeze(1).repeat(1, Wh.size(1), 1, 1)  # [B, 1, N, F'] -> [B, N, N, F']
        a_input = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=-1)  # [B, N, N, 2F']

        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # [B, N, N, 2F'] x [2F', 1] -> [B, N, N]
        attention = F.softmax(e, dim=-1)  # [B, N, N] 归一化注意力权重
        h_prime = torch.bmm(attention, Wh)  # [B, N, N] x [B, N, F'] -> [B, N, F']
        return h_prime


class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.1):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features) for _ in range(heads)
        ])

        self.W = nn.Parameter(torch.Tensor(in_features * heads + in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, h, adj):
        head_outputs = [head(h, adj) for head in self.attention_heads]
        h_prime = torch.cat(head_outputs, dim=-1)
        h_prime = torch.cat([h, h_prime], dim=-1)  # 拼接原始输入
        h_prime = self.dropout(h_prime)
        h_prime = torch.matmul(h_prime, self.W)
        return h_prime


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_layers=2):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gc_layers = nn.ModuleList()

        # 添加一个线性层来调整 residual 的维度
        self.residual_transform = nn.Linear(nfeat, nhid)  # 用于调整残差的维度

        for i in range(num_layers):
            in_features = nfeat if i == 0 else nhid
            out_features = nhid if i < num_layers - 1 else nout
            self.gc_layers.append(GraphConvolution(in_features, out_features, mat_path))

        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(nhid) for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.residual_transform(x)  # 调整 residual 的维度
        for i in range(self.num_layers):
            x = self.gc_layers[i](x)
            if i < self.num_layers - 1:
                x = x.transpose(1, 2).contiguous()
                x = self.bn_layers[i](x).transpose(1, 2).contiguous()
                x = F.relu(x)
                x = self.dropout(x)
        x += residual  # 现在 residual 和 x 的维度应该匹配
        return x, self.gc_layers[-1].adj


class AUwGCN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        mat_dir = '/kaggle/working/ME-GCN-Project'
        mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # 使用GCN来处理输入的节点特征
        self.graph_embedding = GCN(2, 32, 32, mat_path, dropout=0.1, num_layers=2)

        # 多头注意力层
        self.attention = MultiHeadGraphAttentionLayer(in_features=32, out_features=32, heads=4, dropout=0.1)

        self._sequential = nn.Sequential(
            nn.Conv1d(384, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self._classification = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )

        self._init_weight()

    def forward(self, x):
        # 保持三个维度 (batch_size, num_nodes, num_features)
        # 调试
        print(f"Input shape: {x.shape}")  # 调试：输出x的形状
        b, n, c = x.shape  # batch_size, num_nodes, num_features

        # 处理每个时间步的图卷积和注意力层
        x, adj = self.graph_embedding(x)  # (batch_size, num_nodes, feature_size)
        x = self.attention(x, adj)

        # 如果你有多个时间步信息，可以在这里处理多个时间步
        # 假设 x 是 (batch_size, num_nodes, feature_size)，我们只关心对每个时间步的处理

        # 对 x 进行形状调整后，进入卷积层
        x = x.transpose(1, 2)  # (batch_size, feature_size, num_nodes)
        x = self._sequential(x)

        # 分类层
        x = self._classification(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

