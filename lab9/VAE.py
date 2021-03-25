import torch
from torch import nn
from torch.nn import functional as F
class VAEConv2(nn.Module):
    def __init__(self):
        super(VAEConv2, self).__init__()

        # for encoder
        self.fc1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        self.fc21 = nn.Linear(4608//2, 20)
        self.fc22 = nn.Linear(4608//2, 20)

        # for decoder
        self.decoder_linear = torch.nn.Linear(20, 1024*4*4)
        
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.decoder_out = torch.nn.Sigmoid()        

    def encode(self, x):
        x = self.fc1(x)
        h1 = F.relu(x).view(-1,4608//2)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # 0.5 for square root (variance to standard deviation)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self,z):
        # Project and reshape
        x = self.decoder_linear(z)
        x = x.view(-1, 1024, 4, 4)
        x = self.decoder_conv1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_conv3(x)
        x = self.decoder_conv4(x)
        x = self.decoder_out(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # for encoder
        self.fc1 = nn.Linear(3*64*64, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        # for decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 3*64*64)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # 0.5 for square root (variance to standard deviation)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1,3 * 64 * 64))
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar
