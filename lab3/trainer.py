import torch
import time
import os
# from copy import copy
from copy import deepcopy
class trainer():
    def __init__(self,device, criterion, optimizer,scheduler=None) -> None:
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._train_loss = []
        self._train_acc = []
        self._val_loss = []
        self._val_acc = []
        self._init_folder()

    def _init_folder(self) -> None:
        from datetime import datetime
        t = datetime.now().strftime("%Y%m%d-%H%M%S")
        _save_path = f"result-{t}"
        if(os.path.exists(f"{_save_path}") == False):
            os.mkdir(_save_path)
        self._save_path = f"{_save_path}"

    def save(self, weights_name):
        import pickle
        with open(f"{self._save_path}/{weights_name}_val_acc.txt", "wb") as f:
            pickle.dump(self._val_acc, f)
        with open(f"{self._save_path}/{weights_name}_val_loss.txt", "wb") as f:
            pickle.dump(self._val_loss, f)
        with open(f"{self._save_path}/{weights_name}_train_acc.txt", "wb") as f:
            pickle.dump(self._train_acc, f)
        with open(f"{self._save_path}/{weights_name}_train_loss.txt", "wb") as f:
            pickle.dump(self._train_loss, f)

    def train(self, model, dataloaders, num_epochs=25, weights_name='weight_save', is_inception=False):
        device = self.device
        criterion = self.criterion
        optimizer = self.optimizer
        since = time.time()

        self._train_loss = []
        self._train_acc = []
        self._val_loss = []
        self._val_acc = []

        best_model_wts = deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            epoch_start = time.time()
            print(f"Epoch {epoch}/{num_epochs - 1}:LR: {self.optimizer.param_groups[0]['lr']}")
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    # for process anything, device and dataset must put in the same place.
                    # If the model is in GPU, input and output must set to GPU
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    # it uses for update training weights
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            # print('outputs', outputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                epoch_end = time.time()
                
                elapsed_epoch = epoch_end - epoch_start

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                print("Epoch time taken: ", elapsed_epoch)

                # deep copy the model
                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = deepcopy(model.state_dict())
                        torch.save(model.state_dict(), f"{self._save_path}/{weights_name}.pth")
                        self.save(weights_name)
                    self._val_loss.append(epoch_loss)
                    self._val_acc.append(epoch_acc)
                    if(self.scheduler != None):
                        self.scheduler.step(epoch_loss)
                elif phase == 'train':
                    self._train_loss.append(epoch_loss)
                    self._train_acc.append(epoch_acc)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        # load best model weights
        model.load_state_dict(best_model_wts)
        self.save(weights_name)
        return model

    def test(self, model, dataloader, classes):
        import numpy as np
        device = self.device
        model.eval()
        # classes = np.array(('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'))
        #Testing Accuracy
        correct = 0
        total = 0
        #Testing classification accuracy for individual classes.
        class_correct = np.zeros(10)
        class_total = np.zeros(10)
        with torch.no_grad():
            for data in dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                # print(labels)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        print('='*5,"Testing","="*5)
        print('Accuracy of the network on the',total,'test images: %d %%' % (100 * correct / total))
        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))