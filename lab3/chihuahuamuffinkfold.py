import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
from copy import copy
from copy import deepcopy
import torch.nn.functional as F

from sklearn.model_selection import KFold
import numpy as np

# Allow augmentation transform for training set, no augementation for val/test set
# Normalize(mean, std, inplace=False)
# mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

preprocess_augment = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])


full_train_dataset = torchvision.datasets.ImageFolder('data/kaggle')
index = np.arange(16)
kf = KFold(n_splits = 8, shuffle=True)#, random_state=0)
iterator = kf.split(index)

BATCH_SIZE=4
NUM_WORKERS=2
# train_index, val_index = iterator.__next__()

count = 0
best_acc = []
for train_index, val_index in iterator:
    print(f"!!!!!!!!!!!!!!!!!!!!!!! Fold-{count+1} !!!!!!!!!!!!!!!!!!!!")
    train_dataset = torch.utils.data.Subset(full_train_dataset, np.append(train_index,train_index+16))
    train_dataset.dataset = copy(full_train_dataset)
    train_dataset.dataset.transform = preprocess_augment
    val_dataset = torch.utils.data.Subset(full_train_dataset, np.append(val_index, val_index+16))
    val_dataset.dataset.transform = preprocess
    # test_dataset = torchvision.datasets.ImageFolder('data/test', transform=preprocess)
    # print(test_dataset)
    # test_dataloader   = torch.utils.data.DataLoader(test_dataset  , batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_WORKERS)
    # train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [1600, 1988-1600])
    # DataLoaders for the three datasets
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True , num_workers=NUM_WORKERS)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset  , batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_WORKERS)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    from myNetwork.myResNet import ResNet
    from trainer import trainer

    def SEResNet18(num_classes = 10):
        return ResNet(ResNet._BLOCK_SEBASIC, [2, 2, 2, 2], num_classes)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SEResNet18()
    model.load_state_dict(torch.load('result-20210129-104338/seresnet18_adam_0.01.pth'))
    model.classifier[2] = nn.Linear(512,2)
    model.eval()

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    params_to_update = model.parameters()
    # Now we'll use Adam optimization
    optimizer = optim.Adam(params_to_update, lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5)
    t = trainer(device,criterion, optimizer,scheduler)
    model = t.train(model, dataloaders, num_epochs=10, weights_name='seresnet18_chihuahua_muffin_adam_kaggle_0.01')
    count = count + 1
    best_acc.append(t._best_val_acc)

print(best_acc)

