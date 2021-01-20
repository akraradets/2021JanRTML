import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import pickle

# 1. Load CIFAR10
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 5. Use some data augmentation for the training set -- RandomCrop(224) and RandomHorizontalFlip() seem to help.
# 6. Normalize the images to the [-1,1] range using Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if you're training from scratch, or if you're fine tuning the ImageNet weights, you should probably use the magic normalizer of Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), though this can give pixel values outside the [-1,1] range.
preprocess = transforms.Compose([
    transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 1. Use the torchvision CIFAR10 dataset to download the data and load images into memory
# 2. Use PyTorch Dataloaders to shuffle the data and break them into batches during training
# 3. Use a batch size of 200 or less so that you're not using more than 3 GB of GPU RAM (save some for others!)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=0)

classes = np.array(('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))

# 2. Load AlexNet
# https://github.com/pytorch/vision/releases
# The newest version isa 0.8.2 and load empty model
model = torch.hub.load('pytorch/vision:v0.8.2', 'alexnet', pretrained=False)
# print(model.eval())

# https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/
#Updating the second classifier
# AlexNet_model.classifier[4] = nn.Linear(4096,1024)
#Updating the third and the last classifier that is the output layer of the network. Make sure to have 10 output nodes if we are going to get 10 class labels through our model.
model.classifier[6] = nn.Linear(4096,10)
# print(model.eval())

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

#Loss
criterion = nn.CrossEntropyLoss()

#Optimizer(SGD)
# 4. For the optimizer, use stochastic gradient descent with a learning rate around 0.01 or 0.02 to start with. I couldn't get the Adam optimizer working in this experiment at all.
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def test():
    correct_test = 0
    total_test = 0
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
        
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
    return correct_test / total_test, correct_train / total_train 

losses = []
counter = 0
len_batch = len(trainloader)
# 7. Use tensorboard to monitor training, it's awesome!
writer = SummaryWriter()
for epoch in range(300):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # Log input to tensorboard
        # grid = torchvision.utils.make_grid(inputs)
        # writer.add_image('images', grid, i)
        # writer.add_graph(model, images)
        # writer.close()
        
        # print(counter)
        # if(i % 10 == 0):
        #     print("i")
        #     raise ValueError(f"stop ka")
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        writer.add_scalar('pureLoss/train', loss.item(), counter)
        counter = counter + 1
    # Test every epoch
    test_score, train_score = test()
    # Save
    torch.save(model.state_dict(), (f'checkpoints/alexnet-cifar-10-{epoch}-sgd-0.001.pth'))
    print(f"Epoch:{epoch}|Loss:{running_loss} {running_loss / len_batch}|Test:{test_score}|Train:{train_score}")
    writer.add_scalar('EpochLoss/train', running_loss / len_batch, epoch)
    writer.add_scalar('EpochScore/test', test_score * 100, epoch)
    writer.add_scalar('EpochScore/train', train_score * 100, epoch)
    writer.close()


print('Finished Training of AlexNet')
# save looses
with open("losses.txt", "wb") as f:   #Pickling
    pickle.dump(losses, f)
# save model
torch.save(model, "alextrain.model")