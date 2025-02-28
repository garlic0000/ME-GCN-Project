import torch
from model2.model_97 import GraphConvolution
from model2.model_97 import MultiHeadGraphAttentionLayer
from model2.model_97 import TCNBlock


# refer to https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
def configure_optimizers(model, learning_rate, weight_decay):
    """
    This function separates model parameters into two groups:
    - Parameters that will experience weight decay (for regularization)
    - Parameters that will not (e.g., biases, layernorm, embedding weights)
    Then, it returns a PyTorch optimizer object.
    """

    # Initialize two sets: one for parameters that will decay, another for those that won't
    decay = set()  # Parameters to undergo weight decay (regularization)
    no_decay = set()  # Parameters that will not undergo weight decay

    # Whitelist of modules for which we will apply weight decay (e.g., Linear, Conv1d, GraphConvolution, etc.)
    whitelist_weight_modules = (
        torch.nn.Linear, torch.nn.Conv1d, GraphConvolution, MultiHeadGraphAttentionLayer, torch.nn.Conv2d
    )

    # Blacklist of modules for which we will NOT apply weight decay (e.g., BatchNorm layers)
    blacklist_weight_modules = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)

    # Traverse through the model's named modules and parameters
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn  # Full parameter name

            # Check for bias parameters, which don't undergo weight decay
            if pn.endswith('bias'):
                no_decay.add(fpn)
            # Check for BatchNorm layers' weights (they won't undergo weight decay)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
            # Apply weight decay to parameters of modules in the whitelist
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            # Special handling for 'adj' parameters in GraphConvolution, MultiHeadGraphAttentionLayer, etc.
            elif 'adj' in fpn or 'graph_embedding.adj' in fpn:
                decay.add(fpn)
            # Special handling for residual weight parameters, such as 'residual_weight.alpha'
            elif "residual_weight.alpha" in fpn:
                no_decay.add(fpn)
            # Apply weight decay to attention parameters of MultiHeadGraphAttentionLayer
            elif isinstance(m, MultiHeadGraphAttentionLayer) and (pn == 'a' or pn == 'W'):
                decay.add(fpn)
            # Special handling for certain layers like `gat1.a`, `gat.a`, etc.
            elif 'gat1.a' in fpn or 'gat.a' in fpn:
                no_decay.add(fpn)

    # Get all model parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # Ensure no parameters are mistakenly assigned to both decay/no_decay
    inter_params = decay & no_decay
    union_params = decay | no_decay

    # Assert that no parameters exist in both sets
    assert len(inter_params) == 0, (
        f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    )

    # Assert that all parameters are assigned to either decay or no_decay
    assert len(param_dict.keys() - union_params) == 0, (
        f"parameters {str(param_dict.keys() - union_params)} "
        "were not separated into either decay/no_decay set!"
    )

    # Create the optimizer with weight decay applied to the decay set and no decay for the no_decay set
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

    # Instantiate the AdamW optimizer
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer

