# coding=utf-8
import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=False)
        vgg.load_state_dict(torch.load('vgg16-397923af.pth'))
        vgg_loss = nn.Sequential(*list(vgg.features)[:9]).eval()
        # vgg16 2.2: 2th conv before 2th maxpool
        for param in vgg_loss.parameters():
            param.requires_grad = False
        self.vgg_loss = vgg_loss
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.tv_loss = TVLoss()
        self.bce = torch.nn.BCELoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        ones = 0.9 * torch.ones_like(out_labels)
        adversarial_loss = self.bce(out_labels, ones)

        # Perception Loss
        # the pre-trained VGG uses the input range of [0, 1]
        # perception_loss = self.mse_loss(self.vgg_loss((out_images+1)/2), self.vgg_loss((target_images+1)/2))
        perception_loss = self.mse_loss(self.vgg_loss(out_images), self.vgg_loss(target_images))

        # Image Loss
        image_loss = self.l1_loss(out_images, target_images)
        # image_loss = self.mse_loss(out_images, target_images)

        # TV Loss
        tv_loss = self.tv_loss(out_images)
        # return 0.001 * adversarial_loss + perception_loss + 2e-8 * tv_loss
        # return image_loss + 0.001 * adversarial_loss + 2e-6 * perception_loss + 2e-8 * tv_loss
        return image_loss + 0.001 * adversarial_loss + 2e-6 * perception_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
