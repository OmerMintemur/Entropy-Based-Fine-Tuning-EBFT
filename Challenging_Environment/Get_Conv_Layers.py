import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import models


def get_all_conv_layers(model):
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    return conv_layers


model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
conv_layers = get_all_conv_layers(model)
print(f"Total conv layers (SqueezeNet): {len(conv_layers)}")
for name, layer in conv_layers:
    print(name, layer)
print(f"Total conv layers (SqueezeNet): {len(conv_layers)}")