# -*- coding: utf-8 -*-
from Models import return_resnet18_modified, return_resnet34_modified

# Commented out IPython magic to ensure Python compatibility.
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch.optim as optim
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import v2
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle
import json
import random
import time
from torchvision.models import resnet18


class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_std=0.0):
        self.dataset = dataset
        self.noise_std = noise_std

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.noise_std > 0:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
            image = torch.clamp(image, 0, 1)  # Clamp to valid range

        return image, label

    def __len__(self):
        return len(self.dataset)


layers_to_unfreeze = ['layer1.0.conv1', 'layer1.1.conv1', 'layer2.1.conv2', 'layer2.3.conv1',
                      'layer3.0.conv1', 'layer3.1.conv1', 'layer3.1.conv2','layer3.2.conv1',
                      'layer3.2.conv2', 'layer3.3.conv1', 'layer3.3.conv2', 'layer3.4.conv1',
                      'layer3.4.conv2', 'layer3.5.conv1', 'layer3.5.conv2', 'layer4.0.conv1',
                      'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2', 'layer4.2.conv1']
mappings = [0, 0, 0, 0, 0, 0, 0, 0, 0]
model = return_resnet34_modified(mappings)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

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
data_dir = ''
batch_sizes = [8, 16, 32]
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# For faster training
scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
# Train and Test loss stats during training
noise_levels = [0.1, 0.2, 0.3, 0.4]
for batch_size in batch_sizes:
    for std in noise_levels:
        noisy_train_dataset = NoisyDataset(image_datasets['train'], noise_std=std)
        noisy_test_dataset = NoisyDataset(image_datasets['test'], noise_std=std)

        train_loader = torch.utils.data.DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(noisy_test_dataset, batch_size=batch_size, shuffle=False)

        dataloaders = {'train': train_loader, 'test': test_loader}
        dataset_sizes = {'train': len(noisy_train_dataset), 'test': len(noisy_test_dataset)}

        train_loss_during_training = []
        test_loss_during_training = []

        train_accuracy_during_training = []
        test_accuracy_during_training = []
        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        num_epoch = 30
        for epoch in range(num_epoch):
            begin = time.time()
            print(f'Epoch {epoch}/{num_epoch}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    # zero the parameter gradients
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    train_loss_during_training.append(epoch_loss)
                    train_accuracy_during_training.append(epoch_acc.cpu())
                else:
                    test_loss_during_training.append(epoch_loss)
                    test_accuracy_during_training.append(epoch_acc.cpu())

                # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {time.time() - begin} Second')

        # dictionary of lists
        dict = {'Train loss': train_loss_during_training,
                'Test loss': test_loss_during_training,
                'Train Accuracy': train_accuracy_during_training,
                'Test Accuracy': test_accuracy_during_training}

        df = pd.DataFrame(dict)
        df.to_csv(f"Final_Layer_Tuning_Results_Noise_{std}_batch_size_{batch_size}_Kaggle_ResNet34.csv")

        # Evaluation
        # model = model.to(device)
        model.eval()
        # Load the data one more time, this time batch size bigger

        train_loader = torch.utils.data.DataLoader(noisy_train_dataset, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(noisy_test_dataset, batch_size=128, shuffle=False)

        dataloaders = {'train': train_loader, 'test': test_loader}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
        all_preds_train = []
        all_labels_train = []
        for phase in ['train']:
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                preds = torch.argmax(output, 1)
                all_preds_train.append(preds.cpu())
                all_labels_train.append(labels.cpu())

        all_preds_train = torch.cat(all_preds_train, dim=0)
        all_labels_train = torch.cat(all_labels_train, dim=0)
        all_preds_train = all_preds_train.tolist()
        all_labels_train = all_labels_train.tolist()

        all_preds_test = []
        all_labels_test = []
        for phase in ['test']:
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                output = model(inputs)
                preds = torch.argmax(output, 1)
                all_preds_test.append(preds.cpu())
                all_labels_test.append(labels.cpu())

        all_preds_test = torch.cat(all_preds_test, dim=0)
        all_labels_test = torch.cat(all_labels_test, dim=0)
        all_preds_test = all_preds_test.tolist()
        all_labels_test = all_labels_test.tolist()

        with open(f'predictions_labels_final_layer_fine_tune_{std}_batch_size_{batch_size}_Kaggle_ResNet34.pkl',
                  'wb') as f:
            pickle.dump({
                'all_preds_train': all_preds_train,
                'all_labels_train': all_labels_train,
                'all_preds_test': all_preds_test,
                'all_labels_test': all_labels_test
            }, f)
