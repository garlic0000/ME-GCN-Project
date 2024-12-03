import torch
from model.model_1 import GraphConvolution


# refer to https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
def configure_optimizers(model, learning_rate, weight_decay):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    将模型参数划分为需要进行权重衰减的参数和不需要进行权重衰减的参数
    使用AdamW优化器进行优化
    model:需要优化的模型
    learning_rate:学习率
    weight_decay:权重衰减系数 用于正则化
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()  # 需要正则化 进行权重衰减 比如weight
    no_decay = set()  # 不需要正则化 不用权重衰减 比如bias和BatchNorm层的参数
    # 需要进行权重衰减的模块
    # 包括线性层 conv1d层 自定义图卷积层 Corv2d层
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, GraphConvolution, torch.nn.Conv2d)
    # 不需要进行权重衰减的模块
    # 包括BatchNorm1d batchNorm2d 批归一化层参数不需要进行衰减
    blacklist_weight_modules = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
    # 遍历模型模块和参数 分配到衰减和非衰减参数集合中
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

            # 针对model_1的修改
            # parameters {'graph_embedding.0.gc1.parametrizations.weight.original1', 'graph_embedding.0.gc1.parametrizations.weight.original0'} were not separated into either decay/no_decay set!
            # Check if it's a parameter added by weight_norm (e.g., .original0, .original1)
            if 'parametrizations' in fpn:
                continue  # Skip these parameters

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('adj') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)

            # 针对model_1的修改
            # AssertionError: parameters {'graph_embedding.0.gc1.weight_v', 'graph_embedding.0.gc1.weight_g'} were not separated into either decay/no_decay set!
            # Check if the parameter is part of the weight_norm and classify it properly
            if 'weight_v' in pn or 'weight_g' in pn:
                # weight_v and weight_g should be part of the decay parameters
                decay.add(fpn)

    # 获取模型所有参数
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert len(inter_params) == 0, (
        f"parameters {str(inter_params)} made it into both decay/no_decay sets!")
    assert len(param_dict.keys() - union_params) == 0, (
        f"parameters {str(param_dict.keys() - union_params)} "
        "were not separated into either decay/no_decay set!")

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer
