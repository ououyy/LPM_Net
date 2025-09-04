import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'BCELoss', 'DiceLoss', 'FocalLoss']



class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        bce = F.binary_cross_entropy_with_logits(input, target)

        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice



class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice



class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss



"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss



class FocalBCE_Loss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalBCE_Loss, self).__init__()
        self.FL = FocalLoss(gamma=gamma)  # 你需要定义这个类
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        # self.BCE_loss = BceDiceLoss()

    def forward(self, input1, input2, input3, input4, inputs, targets):
        # 确保 targets 是 [B, 1, H, W]
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # 统一所有预测输出尺寸到 targets 大小
        input1 = F.interpolate(input1, size=targets.shape[2:], mode='bilinear', align_corners=False)
        input2 = F.interpolate(input2, size=targets.shape[2:], mode='bilinear', align_corners=False)
        input3 = F.interpolate(input3, size=targets.shape[2:], mode='bilinear', align_corners=False)
        input4 = F.interpolate(input4, size=targets.shape[2:], mode='bilinear', align_corners=False)
        inputs = F.interpolate(inputs, size=targets.shape[2:], mode='bilinear', align_corners=False)

        # 分别计算 BCE + Focal
        loss1 = 0.5 * self.BCE_loss(input1, targets) + self.dice(input1, targets)
        loss2 = 0.5 * self.BCE_loss(input2, targets) + self.dice(input2, targets)
        loss3 = 0.5 * self.BCE_loss(input3, targets) + self.dice(input3, targets)
        loss4 = 0.5 * self.BCE_loss(input4, targets) + self.dice(input4, targets)
        loss_final = 0.5 * self.BCE_loss(inputs, targets) + self.dice(inputs, targets)


        w1 = torch.softmax(loss1, dim=0)
        w2 = torch.softmax(loss2, dim=0)
        w3 = torch.softmax(loss3, dim=0)
        w4 = torch.softmax(loss4, dim=0)
        w5 = torch.softmax(loss_final, dim=0)

        weight_sum = w1 + w2 + w3 + w4 + w5
        w1 = w1 / weight_sum
        w2 = w2 / weight_sum
        w3 = w3 / weight_sum
        w4 = w4 / weight_sum
        w5 = w5 / weight_sum

        loss = (1+w1) * loss1 + w2 * loss2 + w3 * loss3 + w4 * loss4 + w5 * loss_final


        return loss


