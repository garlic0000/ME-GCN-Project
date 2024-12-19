import torch
from torch.autograd import Variable


class MultiCEFocalLoss_New(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, lb_smooth=0.06,
                 reduction='mean', weight_clip=(0.1, 5.0)):
        super(MultiCEFocalLoss_New, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.class_num = class_num
        self.weight_clip = weight_clip  # 类别权重范围

    def forward(self, predict, target):
        pt = torch.softmax(predict, dim=-1).view(-1, self.class_num)
        class_mask = torch.nn.functional.one_hot(target, self.class_num).view(-1, self.class_num)
        ids = target.view(-1, 1)

        # 动态调整 alpha，限制权重范围
        class_counts = class_mask.sum(dim=0).float()
        inverse_class_counts = 1.0 / (class_counts + 1e-7)
        inverse_class_counts = torch.clamp(inverse_class_counts, min=self.weight_clip[0], max=self.weight_clip[1])
        alpha = inverse_class_counts[ids.view(-1)].view(-1, 1)

        # 标签平滑，仅应用于正类
        if self.lb_smooth > 0:
            smooth_factor = self.lb_smooth / (self.class_num - 1)
            class_mask = class_mask * (1 - self.lb_smooth) + smooth_factor

        positive_probs = (pt * class_mask).sum(-1).view(-1, 1)
        log_p = positive_probs.log()
        loss = -alpha * torch.pow((1 - positive_probs), self.gamma) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def _focal_loss(output, label, gamma, alpha, lb_smooth):
    output = output.contiguous().view(-1)
    label = label.view(-1)
    mask_class = (label > 0).float()

    c_1 = alpha
    c_0 = 1 - c_1
    loss = ((c_1 * torch.abs(label - output) ** gamma * mask_class
             * torch.log(output + 0.00001))
            + (c_0 * torch.abs(label + lb_smooth - output) ** gamma
               * (1.0 - mask_class)
               * torch.log(1.0 - output + 0.00001)))
    loss = -torch.mean(loss)
    return loss


def _probability_loss(output, score, gamma, alpha, lb_smooth):
    output = torch.sigmoid(output)
    loss = _focal_loss(output, score, gamma, alpha, lb_smooth)
    return loss
