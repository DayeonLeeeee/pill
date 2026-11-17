"""Independent hierarchical classifiers, initialized without any checkpoint.

Training consumes logits with cross_entropy. Softmax is applied for inference.
Color uses one categorical output over single colors and unordered two-color sets.
"""
import torch
from torch import nn
from torchvision.models import resnet18


class HierarchicalClassifier(nn.Module):
    initialization = 'random; weights=None; no original pill checkpoint'

    def __init__(self, num_classes, channels):
        super().__init__()
        self.channels = channels
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(channels, 64, 7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.LayerNorm(128+256+512), nn.Dropout(.2),
                                        nn.Linear(128+256+512, num_classes))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'Expected N x {self.channels} x H x W input')
        x = self.stem(x)
        hierarchy = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index >= 1:
                hierarchy.append(self.pool(x).flatten(1))
        return self.classifier(torch.cat(hierarchy, dim=1))

    @torch.inference_mode()
    def probabilities(self, x):
        if self.training:
            raise RuntimeError('Call model.eval() before inference')
        return self(x).softmax(dim=-1)


class ShapeClassifier(HierarchicalClassifier):
    input_mode = 'L'

    def __init__(self, num_classes):
        super().__init__(num_classes, 1)


class ColorClassifier(HierarchicalClassifier):
    input_mode = 'RGB'

    def __init__(self, num_classes):
        super().__init__(num_classes, 3)
