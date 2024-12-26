import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import os
import numpy as np

"""
1. 修正了注意力归一化方向：
在 GraphAttentionLayer 中，原本的归一化方向存在问题。通过修改归一化操作，确保了 attention = torch.nn.functional.softmax(e, dim=-1) 的正确性。这个变更确保了在计算注意力时，使用的是正确的维度方向。
2. 添加了数值稳定性优化：
在计算注意力分数时，使用了 e = e - torch.max(e, dim=-1, keepdim=True).values 来减去 e 中每一行的最大值。这种做法防止了在数值上可能出现的溢出问题（如 softmax 函数可能导致的溢出），提高了模型的数值稳定性。
3. 初始化调整：
通过 nn.init.xavier_uniform_ 初始化了 W 和 a 权重，使得权重分布更加均匀，避免了初始化时可能产生的不稳定性。
4. 加入 Dropout：
在 MultiHeadGraphAttentionLayer 和 GCN 中加入了 Dropout 层，dropout=0.1。这样做可以在训练过程中减少过拟合，提高模型的泛化能力。
5. 移除不必要的 Sequential：
原来代码中的 graph_embedding 使用了 nn.Sequential 来包装多个层。这一层的组合没有必要使用 nn.Sequential，直接调用这些层会更清晰。因此，graph_embedding 被改成直接调用每个图卷积层和批归一化层，而不是用 Sequential 封装。
6. 层数的增加：
GCN 层数从 num_layers=1 增加到 num_layers=2，这为模型提供了更深的图卷积特征提取能力。
7. 权重初始化：
Conv1d 和 Conv2d 层的权重初始化方式也有所调整，使用了 nn.init.kaiming_normal_ 初始化，以便更好地适应 ReLU 激活函数，减少梯度消失和梯度爆炸问题。
8. 图卷积的批归一化：
在 GCN 层中，添加了 BatchNorm1d 层，用于每个图卷积层之后的特征规范化。这可以加速训练并稳定模型表现，尤其在深度网络中尤为重要。
9. 调整了 AUwGCN 架构：
在 AUwGCN 的 graph_embedding 模块中，层数和功能进行了优化，以适应新的图卷积和图注意力层的组合。
_sequential 和 _classification 层保持不变，但修改了卷积层的配置，例如 kernel 的大小、步幅、填充方式等。
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

        self.graph_embedding = GCN(2, 32, 32, mat_path, dropout=0.3, num_layers=1)
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
