# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:22:25 2024

@author: OMER
"""

import torch
import torch.nn as nn

def return_resnet18_modified(mappings=None):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
    for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    return model

def return_resnet34_modified(mappings=None):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights='ResNet34_Weights.DEFAULT')
    for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    return model