from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import pprint
from torchvision import transforms, datasets
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from Models import return_resnet18_modified
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

layers_name = ['layer1.0.conv1', 'layer1.0.conv2',
               'layer1.1.conv1', 'layer1.1.conv2',
               'layer2.0.conv1', 'layer2.0.conv2',
               'layer2.0.downsample.0', 'layer2.0.downsample.1',
               'layer2.1.conv1', 'layer2.1.conv2',
               'layer3.0.conv1', 'layer3.0.conv2',
               'layer3.0.downsample.0', 'layer3.0.downsample.1',
               'layer3.1.conv1', 'layer3.1.conv2',
               'layer4.0.conv1', 'layer4.0.conv2',
               'layer4.0.downsample.0', 'layer4.0.downsample.1',
               'layer4.1.conv1', 'layer4.1.conv2']


def calculate_entropy(image):
    '''Function will calculate the entropy of a given image'''
    temp = np.floor(image * 256)
    temp_int = temp.clone().detach()
    # temp_int = torch.tensor(temp, dtype=torch.int32)
    temp_images = torch.clamp(temp_int, 0, 255)

    M = temp_images.numpy()
    hist, _ = np.histogram(M.ravel(), bins=256, range=(0, 256))
    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist, base=2)
    return image_entropy


# Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load the data
data_dir = 'Brain Tumor MRI Dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
print(image_datasets)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True) for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
# class_names = image_datasets['train'].classes

model_resnet = return_resnet18_modified()  # Get the Model
# This is just for printing the layers
# ------------------------------------
# train_nodes, eval_nodes = get_graph_node_names(model_resnet)
# pprint.pprint(eval_nodes)
# ------------------------------------
feat_ext = create_feature_extractor(model_resnet, return_nodes=layers_name)
entropies_for_graph = {}
a = 0
for datax in ['train']:
    for layer in layers_name:  # Traverse all layers
        entropy_temp = 0
        for i, data in enumerate(dataloaders[datax], 0):  # Batch size is 1, so traverse image by image
            image, labels = data
            image, labels = image.to(device), labels.to(device)
            with torch.no_grad():
                feat_ext = feat_ext.to(device)
                out = feat_ext(image)  # Get the features

            how_many_channel = list(out[layer].shape)[1]  # Ex: torch.Size([1, 64, 56, 56])

            for channel in range(0, how_many_channel):  # For each channel traverse and calculate entropy
                entropy_temp += calculate_entropy(out[layer][0, channel, :, :].cpu().detach())


        # entropies.append(entropy_temp/how_many_channel)
        print(f'Calculation is done for Layer {layer}')

        entropies_for_graph[layer] = entropy_temp / how_many_channel  # Normalize
        print(entropy_temp / how_many_channel)
print(entropies_for_graph)

with open('Results.txt', 'w') as convert_file:
    convert_file.write(json.dumps(entropies_for_graph))
