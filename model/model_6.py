import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, mat_path, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        adj_mat = np.load(mat_path)
        self.register_buffer('adj', torch.from_numpy(adj_mat).float())

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        b, n, f = input.shape  # B: batch size, N: nodes, F: features
        print(f"Before reshape: input shape = {input.shape}")

        # 检查输入特征维度是否与预期的 in_features 一致
        if f != self.in_features:
            print(f"Adjusting input shape from {f} to {self.in_features}")
            if f > self.in_features:
                # 如果输入特征数过多，进行切割，保留前 in_features 个特征
                input = input[:, :, :self.in_features]
            else:
                # 如果输入特征数过少，通过填充补齐到 in_features 维度
                padding = self.in_features - f
                input = F.pad(input, (0, padding), "constant", 0)

        print(f"After reshape: input shape = {input.shape}")

        weight = self.weight.unsqueeze(0).repeat(b, 1, 1)  # 扩展到每个批次
        print(f"Weight shape: {weight.shape}")

        support = torch.bmm(input, weight)  # 批量矩阵乘法
        print(f"Support shape: {support.shape}")

        output = torch.bmm(self.adj.unsqueeze(0).repeat(b, 1, 1), support)  # 与邻接矩阵乘法

        if self.bias is not None:
            return output + self.bias
        else:
            return output



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, mat_path, dropout=0.3, num_layers=2):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.gc_layers = nn.ModuleList()

        for i in range(num_layers):
            in_features = nfeat if i == 0 else nhid
            out_features = nhid if i < num_layers - 1 else nout
            self.gc_layers.append(GraphConvolution(in_features, out_features, mat_path))

        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(nhid) for _ in range(num_layers - 1)])

        # 输入维度调整为192（根据实际需求）
        self.adjust_input = nn.Linear(24, 192)  # 以前是 6480 -> 24

        # 使用卷积层来处理调整后的输入
        self.conv1d_layer = nn.Conv1d(in_channels=192, out_channels=64, kernel_size=1)

    def forward(self, x):
        residual = x  # 保存输入用于残差连接

        # 打印输入的形状
        print("Before adjust_input:", x.shape)

        b, n, f, _ = x.shape  # 假设 extra_dim 是额外的维度
        x = x.view(b, n, -1)  # 将 extra_dim 展开，变为 [batch_size, nodes, features * extra_dim]

        print("After flattening:", x.shape)

        x = self.adjust_input(x)  # 调整输入通道数为 192
        print("After adjust_input:", x.shape)

        # 转换形状为适应 Conv1d
        x = x.permute(0, 2, 1)  # 变为 [batch_size, channels, length]
        print("After permute:", x.shape)

        # 经过卷积层
        x = self.conv1d_layer(x)
        print("After conv1d:", x.shape)

        # 进入图卷积层
        for i in range(self.num_layers):
            print(f"Before gc_layer {i}:", x.shape)
            x = self.gc_layers[i](x)

            if i < self.num_layers - 1:
                x = x.transpose(1, 2).contiguous()
                x = self.bn_layers[i](x).transpose(1, 2).contiguous()
                x = F.relu(x)

            print(f"After gc_layer {i}:", x.shape)

        if residual.shape[-1] != x.shape[-1]:
            residual = self._adjust_residual(residual, x.shape[-1], x.device)

        return x + residual

    def _adjust_residual(self, residual, target_dim, device):
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

        self._classification = torch.nn.Conv1d(64, 3 + 3 + 2 + 2, kernel_size=3, stride=1, padding=2)

    def forward(self, input_data):
        residual = input_data
        # 输入数据的处理和前馈网络计算
        x = self.graph_embedding(input_data)
        x = self.attention(x, self.graph_embedding[0].adj)  # 图注意力
        x = self._sequential(x)
        x = self._classification(x)

        return x + residual

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
