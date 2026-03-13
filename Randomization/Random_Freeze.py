from Models import return_resnet18_modified, return_resnet34_modified
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import torch.nn.functional as F
import time
import random

# Labels dictionary
labels_ = {
    0: 'glioma',
    1: 'meningioma',
    2: 'notumor',
    3: 'pituitary'
}


# Random freeze function
def random_freeze_layers(model, freeze_prob=0.5):
    """
    Randomly freeze layers with probability freeze_prob.
    Return list of unfrozen layer names.
    """
    unfrozen_layers = []
    for name, param in model.named_parameters():
        if random.random() > freeze_prob:
            param.requires_grad = True
            unfrozen_layers.append(name)
        else:
            param.requires_grad = False
    return unfrozen_layers


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data_dir = ''
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

# Set batch size you want to test with:
BATCH_SIZE = 16
dataset_ = "Kaggle"
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) for x in
               ['train', 'test']}

num_iterations = 5
num_epochs = 30
freeze_prob = 0.5  # Probability to freeze each layer

for iteration in range(num_iterations):
    print(f"\n--- Iteration {iteration + 1} / {num_iterations} ---")

    # Initialize model
    model = return_resnet18_modified()

    # Randomly freeze/unfreeze layers
    unfrozen_layers = random_freeze_layers(model, freeze_prob=freeze_prob)
    print(f"Number of unfrozen layers: {len(unfrozen_layers)}")
    # print("Unfrozen layers:")
    for layer_name in unfrozen_layers:
        print(layer_name)

    model = model.to(device)

    # Setup optimizer only for unfrozen parameters
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=0.0001)

    criterion = nn.CrossEntropyLoss()
    # Train and Test loss stats during training
    train_loss_during_training = []
    test_loss_during_training = []

    train_accuracy_during_training = []
    test_accuracy_during_training = []
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epoch = 30
    time_begin = time.time()
    for epoch in range(num_epoch):
        # print(f'Epoch {epoch}/{num_epoch}')
        # print('-' * 10)

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
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

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

            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    time_end = time.time()

    # dictionary of lists
    dict = {'Train loss': train_loss_during_training,
            'Test loss': test_loss_during_training,
            'Train Accuracy': train_accuracy_during_training,
            'Test Accuracy': test_accuracy_during_training}

    # Evaluation

    model.eval()
    # Load the data one more time, this time batch size bigger
    data_dir = ''
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True) for x in
                   ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    all_preds_train = []
    all_labels_train = []
    time_begin = time.time()
    for phase in ['train']:
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            preds = torch.argmax(output, 1)
            all_preds_train.append(preds.cpu())
            all_labels_train.append(labels.cpu())
    time_end = time.time()
    # print(f"BATCH SIZE is : {BATCH_SIZE} -- Time elapsed during trainig is {time_end - time_begin:.2f}")
    all_preds_train = torch.cat(all_preds_train, dim=0)
    all_labels_train = torch.cat(all_labels_train, dim=0)
    all_preds_train = all_preds_train.tolist()
    all_labels_train = all_labels_train.tolist()

    all_preds_test = []
    all_labels_test = []
    # Initialize lists to store true labels and probabilities for test set
    all_probs_test = []
    all_paths_test = []
    idx = 0
    for phase in ['test']:
        for inputs, labels in dataloaders[phase]:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            output = model(inputs)

            # get file paths from the dataset, using idx to idx+batch_size
            batch_paths = [image_datasets['test'].imgs[i][0] for i in range(idx, idx + batch_size)]

            # Convert outputs to probabilities using softmax
            probs = F.softmax(output, dim=1)  # Apply softmax to get probabilities for each class

            # Store the probabilities and true labels
            all_probs_test.append(probs.cpu().detach())  # Detach to avoid tracking in the computation graph

            preds = torch.argmax(output, 1)
            all_preds_test.append(preds.cpu())
            all_labels_test.append(labels.cpu())
            all_paths_test.append(batch_paths)

            idx += batch_size

    # model_save_path = f'resnet18_final_layer_fine_tuned_batch_{BATCH_SIZE}.pth'
    # torch.save(model.state_dict(), model_save_path)

    all_preds_test = torch.cat(all_preds_test, dim=0)
    all_labels_test = torch.cat(all_labels_test, dim=0)
    all_probs_test = torch.cat(all_probs_test, dim=0)
    all_preds_test = all_preds_test.tolist()
    all_labels_test = all_labels_test.tolist()

    # Save unfrozen layers info
    with open(f"ResNet18_unfrozen_layers_iteration_{iteration + 1}_{dataset_}.txt", "w") as f:
        for layer in unfrozen_layers:
            f.write(f"{layer}\n")

    with open(f'ResNet18_Random_{iteration}_{dataset_}.pkl', 'wb') as f:
        pickle.dump({
            'all_preds_train': all_preds_train,
            'all_labels_train': all_labels_train,
            'all_preds_test': all_preds_test,
            'all_labels_test': all_labels_test,
            'all_probps_test': all_probs_test
        }, f)
