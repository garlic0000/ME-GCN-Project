import torch
from torch.autograd import Variable
import torch.nn.functional as F


class MultiCEFocalLoss_New(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, lb_smooth=0.06,
                 reduction='mean', weight_clip=(0.1, 5.0)):
        """
        Multi-class Focal Loss with Label Smoothing and Dynamic Class Weights.

        Args:
            class_num (int): Number of classes.
            gamma (float): Focusing parameter for Focal Loss. Default is 2.
            alpha (torch.Tensor or None): Class weights. If None, uniform weights are used.
            lb_smooth (float): Label smoothing factor. Default is 0.06.
            reduction (str): Reduction method. 'mean' or 'sum'. Default is 'mean'.
            weight_clip (tuple): Min and max bounds for class weights. Default is (0.1, 5.0).
        """
        super(MultiCEFocalLoss_New, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))  # Default uniform weights
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.class_num = class_num
        self.weight_clip = weight_clip  # Class weight range

    def forward(self, predict, target):
        """
        Forward pass of the loss function.

        Args:
            predict (torch.Tensor): Model predictions (logits), shape [B, N, C].
            target (torch.Tensor): Ground truth labels, shape [B, N].

        Returns:
            torch.Tensor: Computed loss.
        """
        # Convert logits to probabilities
        pt = F.softmax(predict, dim=-1).view(-1, self.class_num)  # Shape [B*N, C]
        target_onehot = F.one_hot(target, num_classes=self.class_num).float()  # One-hot encoding [B*N, C]

        # Compute dynamic class weights
        class_counts = target_onehot.sum(dim=0) + 1e-7  # Class occurrence counts [C]
        inverse_class_weights = 1.0 / class_counts  # Inverse of class frequency
        clipped_weights = torch.clamp(inverse_class_weights, min=self.weight_clip[0], max=self.weight_clip[1])
        alpha = clipped_weights[target.view(-1)]  # Select weights for each sample [B*N]

        # Apply label smoothing if enabled
        if self.lb_smooth > 0:
            smooth_factor = self.lb_smooth / (self.class_num - 1)
            target_onehot = target_onehot * (1 - self.lb_smooth) + smooth_factor

        # Compute positive probabilities
        positive_probs = (pt * target_onehot).sum(-1)  # P_t [B*N]

        # Compute focal loss
        log_p = positive_probs.log()  # log(P_t)
        focal_loss = -alpha * torch.pow(1 - positive_probs, self.gamma) * log_p  # Focal loss [B*N]

        # Apply reduction method
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss


def _focal_loss(output, label, gamma, alpha, lb_smooth):
    """
    Binary Focal Loss function for auxiliary probability loss.

    Args:
        output (torch.Tensor): Predicted probabilities (logits).
        label (torch.Tensor): Ground truth labels (binary).
        gamma (float): Focusing parameter for Focal Loss.
        alpha (float): Weight for positive class.
        lb_smooth (float): Label smoothing factor.

    Returns:
        torch.Tensor: Computed binary focal loss.
    """
    output = torch.sigmoid(output).view(-1)
    label = label.view(-1)
    mask_class = (label > 0).float()  # Mask for positive samples

    c_1 = alpha  # Positive class weight
    c_0 = 1 - c_1  # Negative class weight

    # Compute loss for positive and negative samples
    positive_loss = c_1 * torch.pow(torch.abs(label - output), gamma) * mask_class * torch.log(output + 1e-6)
    negative_loss = c_0 * torch.pow(torch.abs(label + lb_smooth - output), gamma) * (1.0 - mask_class) * torch.log(
        1.0 - output + 1e-6)

    # Combine losses
    loss = -torch.mean(positive_loss + negative_loss)
    return loss


def _probability_loss(output, score, gamma, alpha, lb_smooth):
    """
    Wrapper for binary focal loss with sigmoid activation.

    Args:
        output (torch.Tensor): Predicted logits.
        score (torch.Tensor): Ground truth scores.
        gamma (float): Focusing parameter for Focal Loss.
        alpha (float): Weight for positive class.
        lb_smooth (float): Label smoothing factor.

    Returns:
        torch.Tensor: Computed probability loss.
    """
    output = torch.sigmoid(output)
    loss = _focal_loss(output, score, gamma, alpha, lb_smooth)
    return loss
