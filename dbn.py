import torch
import torch.nn as nn
from torchvision import models

BACKBONES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "efficientnet_b0": models.efficientnet_b0,
}


def _build_backbone(name, pretrained):
    ctor = BACKBONES[name]
    if not pretrained:
        return ctor()
    try:
        return ctor(weights="IMAGENET1K_V1")
    except TypeError:
        return ctor(pretrained=True)


def _strip_head(model):
    if hasattr(model, "fc"):
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return dim
    for module in model.classifier:
        if isinstance(module, nn.Linear):
            dim = module.in_features
            model.classifier = nn.Identity()
            return dim
    raise ValueError("backbone head not found")


class DBN(nn.Module):
    def __init__(self, backbone="resnet50", num_classes=None, pretrained=True):
        super().__init__()
        self.global_branch = _build_backbone(backbone, pretrained)
        self.local_branch = _build_backbone(backbone, pretrained)
        g_dim = _strip_head(self.global_branch)
        l_dim = _strip_head(self.local_branch)
        self.feature_dim = g_dim + l_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes) if num_classes else None

    def forward(self, global_img, local_img):
        gf = self.global_branch(global_img)
        lf = self.local_branch(local_img)
        feat = torch.cat((gf, lf), dim=1)
        if self.classifier is None:
            return feat
        return self.classifier(feat), feat
