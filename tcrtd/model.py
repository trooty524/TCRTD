"""Backbones and the ReID head.

Both backbone families expose a token grid. The descriptor is the normalized
mean token, and the BNNeck classifier uses the same mean token.

    DINOv2-S/14, B/14   448 x 224 input, 32 x 16 patch tokens (384 / 768-d)
    ResNet-50, 101      256 x 128 input, last stride 1, 16 x 8 map (2048-d)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

BACKBONES = {
    # name: (architecture, embedding dimension, input size, token grid)
    "dinov2_s14": ("vit_small_patch14_dinov2", 384, (448, 224), (32, 16)),
    "dinov2_b14": ("vit_base_patch14_dinov2", 768, (448, 224), (32, 16)),
    "resnet50": ("resnet50", 2048, (256, 128), (16, 8)),
    "resnet101": ("resnet101", 2048, (256, 128), (16, 8)),
}
DINOV2_URLS = {
    "dinov2_s14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "dinov2_b14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
}


def is_resnet(name: str) -> bool:
    return name.startswith("resnet")


class ResNetTokens(nn.Module):
    """ImageNet ResNet trunk with a stride-1 final stage."""

    def __init__(self, trunk: nn.Module):
        super().__init__()
        self.stem = nn.Sequential(trunk.conv1, trunk.bn1, trunk.relu, trunk.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = trunk.layer1, trunk.layer2, trunk.layer3, trunk.layer4
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(images)))))
        return x.flatten(2).transpose(1, 2)                      # [B, H*W, C]


class DINOv2Tokens(nn.Module):
    """DINOv2 ViT returning patch tokens without class or register tokens."""

    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features(images)
        return tokens[:, self.vit.num_prefix_tokens:]


def _fix_position_grid(vit: nn.Module, image_size, grid) -> None:
    """Resize the DINOv2 position table to the training grid."""
    from timm.layers.pos_embed import resample_abs_pos_embed

    with torch.no_grad():
        table = resample_abs_pos_embed(vit.pos_embed.detach(), new_size=list(grid),
                                       old_size=list(vit.patch_embed.grid_size),
                                       num_prefix_tokens=vit.num_prefix_tokens)
    vit.pos_embed = nn.Parameter(table)
    vit.dynamic_img_size = False
    vit.patch_embed.img_size, vit.patch_embed.grid_size = tuple(image_size), tuple(grid)
    vit.patch_embed.flatten, vit.patch_embed.strict_img_size = True, True


def build_backbone(name: str, weights: str | None = None, pretrained: bool = True) -> nn.Module:
    """Build a backbone and load local or downloaded pretrained weights."""
    arch, _, image_size, grid = BACKBONES[name]
    if is_resnet(name):
        import torchvision

        if pretrained and weights is None:
            trunk = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1")
        else:
            trunk = getattr(torchvision.models, arch)(weights=None)
            if pretrained:
                trunk.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
        return ResNetTokens(trunk)

    import timm

    vit = timm.create_model(arch, pretrained=False, num_classes=0, dynamic_img_size=True)
    if pretrained:
        state = (torch.hub.load_state_dict_from_url(DINOV2_URLS[name], map_location="cpu")
                 if weights is None else torch.load(weights, map_location="cpu"))
        missing, unexpected = vit.load_state_dict(state, strict=False)
        if missing or set(unexpected) - {"mask_token"}:
            raise RuntimeError(f"unexpected DINOv2 state: missing={missing}, unexpected={unexpected}")
    _fix_position_grid(vit, image_size, grid)
    return DINOv2Tokens(vit)


class ReIDModel(nn.Module):
    """Backbone, mean-token descriptor and BNNeck classifier."""

    def __init__(self, backbone_name: str, num_classes: int, weights: str | None = None,
                 pretrained: bool = True):
        super().__init__()
        _, dim, self.image_size, _ = BACKBONES[backbone_name]
        self.backbone = build_backbone(backbone_name, weights, pretrained)
        self.bnneck = nn.BatchNorm1d(dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, images: torch.Tensor):
        """Return pooled features, normalized descriptors and identity logits."""
        feature = self.backbone(images).float().mean(dim=1)
        return feature, F.normalize(feature, dim=1), self.classifier(self.bnneck(feature))

    @torch.no_grad()
    def describe(self, images: torch.Tensor) -> torch.Tensor:
        """Return the unit descriptor used for cosine matching."""
        return self.forward(images)[1]
