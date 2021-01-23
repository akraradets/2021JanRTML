# https://github.com/dsai-asia/RTML/blob/main/Labs/02-PyTorch-AlexNet-GoogLeNet/02-PyTorch-AlexNet-GoogLeNet.ipynb
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy
import torch.nn.functional as F

preprocess = transforms.Compose([
    transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# Download CIFAR-10 and split into training, validation, and test sets
train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=preprocess)

# Split the training set into training and validation sets randomly.
# CIFAR-10 train contains 50,000 examples, so let's split 80%/20%.
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])

# Download the test set. If you use data augmentation transforms for the training set, you'll want to use a different transformer here.
### If we do data augment, we don't "usually" augment the test data ###
test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False,download=True, transform=preprocess)


train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=4, shuffle=True , num_workers=2)
val_dataloader  = torch.utils.data.DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset,  batch_size=4, shuffle=False, num_workers=2)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

from myModule import GoogLeNet

model = GoogLeNet()
model.eval()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
params_to_update = model.parameters()
optimizer = optim.SGD(params_to_update , lr=0.001, momentum=0.9)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, weights_name='weight_save', is_inception=False):
    since = time.time()

    val_acc_history = []
    loss_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs,aux4a,aux4d = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux4a, labels)
                        loss3 = criterion(aux4d, labels)
                        loss = loss1 + (0.3 * loss2) + (0.3 * loss3)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    # Backpropagate only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # Gather our summary statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_end = time.time()
            
            elapsed_epoch = epoch_end - epoch_start

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print("Epoch time taken: ", elapsed_epoch)

            # If this is the best model on the validation set so far, deep copy it
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), weights_name + ".pth")
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                loss_acc_history.append(epoch_loss)

        print()

    # Output summary statistics, load the best weight set, and return results
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, loss_acc_history

dataloaders = { 'train': train_dataloader, 'val': val_dataloader }
best_model, val_acc_history, loss_acc_history = train_model(model, dataloaders, criterion, optimizer, 50, 'google_scratch_augment_lr_0.001_bestsofar', is_inception=True)

import pickle
with open("google_scratch_augment_val_acc_history.txt", "wb") as f:
    pickle.dump(val_acc_history, f)

with open("google_scratch_augment_loss_acc_history.txt", "wb") as f:
    pickle.dump(loss_acc_history, f)
