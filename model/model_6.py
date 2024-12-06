import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os

"""
关键更改：
在model_5的基础上
通道数不变：
残差连接：避免梯度消失，提升训练效果。
多头注意力机制：增强了模型的稳定性和表达能力。
邻接矩阵正则化：通过自连接和归一化提高了训练的稳定性。
输入数据的形状调整：保证输入数据符合模型的要求 (B, T, N, C)。
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
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))  # no weight_norm
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
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
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_layers=2):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.gc_layers = nn.ModuleList()

        # 添加多个图卷积层
        for i in range(num_layers):
            in_features = nfeat if i == 0 else nhid
            out_features = nhid if i < num_layers - 1 else nout
            self.gc_layers.append(GraphConvolution(in_features, out_features, mat_path))

        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(nhid) for _ in range(num_layers - 1)])

        # 这里确保输入维度匹配
        self.adjust_input = nn.Linear(768, 192)  # 这里的 768 要与输入的第二维匹配

        # 用一个卷积层来处理调整后的输入
        self.conv1d_layer = nn.Conv1d(in_channels=192, out_channels=64, kernel_size=1)

    def forward(self, x):
        residual = x  # 保存输入，用于残差连接

        # 调整输入形状
        x = x.view(x.size(0), -1)  # 将输入展平为 [batch_size, input_features]
        print("Before adjust_input:", x.shape)
        x = self.adjust_input(x)  # 调整输入通道数为 192
        print("After adjust_input:", x.shape)

        # 转换形状为适应 Conv1d
        x = x.unsqueeze(2)  # 添加一个维度，以便与 Conv1d 适配，即 [batch_size, channels, length]
        print("After unsqueeze:", x.shape)

        # 经过卷积层
        x = self.conv1d_layer(x)

        for i in range(self.num_layers):
            x = self.gc_layers[i](x)

            # 如果我们不是最后一层，则进行 BatchNorm 和 ReLU
            if i < self.num_layers - 1:
                x = x.transpose(1, 2).contiguous()
                x = self.bn_layers[i](x).transpose(1, 2).contiguous()
                x = F.relu(x)

        # 确保输出的维度和输入维度一致
        if residual.shape[-1] != x.shape[-1]:
            residual = self._adjust_residual(residual, x.shape[-1], x.device)

        return x + residual, self.gc_layers[-1].adj

    def _adjust_residual(self, residual, target_dim, device):
        # 使用线性层调整 residual 的维度以匹配 target_dim，并确保它在正确的设备上
        linear = nn.Linear(residual.shape[-1], target_dim).to(device)
        return linear(residual)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4):  # 多头注意力
        super(GraphAttentionLayer, self).__init__()
        assert in_features == out_features
        self.num_heads = num_heads
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, h, adj):
        # 通过正则化邻接矩阵
        adj = adj + torch.eye(adj.size(1), device=adj.device)  # 添加自连接
        adj = torch.nn.functional.normalize(adj, p=1, dim=1)  # 正则化

        Wh = torch.matmul(h, self.W)

        # 为每个注意力头计算
        head_outputs = []
        for _ in range(self.num_heads):
            Wh_repeat_1 = Wh.unsqueeze(2).repeat(1, 1, Wh.size(1), 1)
            Wh_repeat_2 = Wh.unsqueeze(1).repeat(1, Wh.size(1), 1, 1)
            a_input = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=-1)

            e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))
            attention = torch.nn.functional.softmax(e, dim=1)

            h_prime = torch.bmm(attention, Wh)
            head_outputs.append(h_prime)

        # 将多个头的输出拼接
        return torch.cat(head_outputs, dim=-1)


class AUwGCN(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        # 服务器测试路径
        mat_dir = '/kaggle/working/ME-GCN-Project'
        mat_path = os.path.join(mat_dir, 'assets', '{}.npy'.format(opt['dataset']))

        # 这里增加了更多的GCN层
        self.graph_embedding = torch.nn.Sequential(GCN(2, 16, 16, mat_path, num_layers=2))

        in_dim = 192  # 保留输入通道数为192

        self.attention = GraphAttentionLayer(in_features=16, out_features=16, num_heads=4)  # 调整 GraphAttentionLayer

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

        self._classification = torch.nn.Conv1d(64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2, dilation=2,
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
