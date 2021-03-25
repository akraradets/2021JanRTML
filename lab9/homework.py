from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VAE import VAEConv2 as VAEConv

import utils
global plotter
plotter = utils.VisdomLinePlotter()


log_interval = 100
seed = 1

torch.manual_seed(seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

out_dir = '../torch_data/DCGAN/celeb/' #you can use old downloaded dataset, I use from VGAN
batch_size=8

compose = transforms.Compose(
        [
            transforms.Resize((64,64)),
            transforms.ToTensor()
            # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])


dataset = datasets.ImageFolder(root=out_dir, transform=compose)
# print(dataset)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [254, 63])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#,  num_workers=1, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)#, num_workers=1, pin_memory=True)



# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # print("recon:",recon_x.shape)
    # print("x:",x.shape)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # print(data.shape)
        data = data.to(device)
        # print("---- a")
        optimizer.zero_grad()
        # print("---- b")
        recon_batch, mu, logvar = model(data)
        # print("---- c")
        loss = loss_function(recon_batch.view(-1, 3, 64,64), data, mu, logvar)
        # print("---- d")
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # print("---- e")
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    # print("---- f")
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    plotter.plot('loss', 'train', 'Loss over epoch', x=epoch, y=train_loss / len(train_loader.dataset))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch.view(-1, 3, 64,64), data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(batch_size, 3, 64,64)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                print(comparison.shape)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    plotter.plot('loss', 'test', 'Loss over epoch', x=epoch, y=test_loss)
    plotter.viz.images(comparison.cpu(), win='reconstrction')

print("to_device")
model = VAEConv().to(device)
print("Optimizer")
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters() , lr=0.00001)

epochs = 100

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        print("save image: " + 'results/sample_' + str(epoch) + '.png')
        save_image(sample.view(-1, 3, 64,64), 'results/sample_' + str(epoch) + '.png')
        plotter.viz.images(sample.view(-1, 3, 64,64), win='sample')