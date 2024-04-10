import torch
import torch.nn as nn
import src.support as support


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)


class Cifar100_nn(nn.Module):

    def __init__(self, in_channels, num_classes, device):
        super(Cifar100_nn, self).__init__()
        self.device = device
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

        self.to(self.device)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

    def detailed_forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out#, output

    def _train_epoch(self, optimizer, train_loader, criterion):
        self.train()
        for _, (data, target, _) in enumerate(train_loader):
            optimizer.zero_grad()
            output = self(data.to(self.device))
            loss = criterion(output.to(self.device), target.to(self.device))
            loss.backward()
            optimizer.step()

    def evaluate(self, criterion, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self(data.to(self.device))
                test_loss += criterion(output.to(self.device), target.to(self.device)).item()
                pred = output.to("cpu").data.max(1, keepdim=True)[1]
                correct += pred.eq(target.to("cpu").data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        return test_loss, (correct / len(test_loader.dataset) * 100)

    def fit(self, epochs, criterion, optimizer, train_loader):
        for epoch in range(epochs):
            self._train_epoch(optimizer, train_loader, criterion)

    def get_embedding_dim(self):
        return 10

    def save(self, path):
        torch.save(self.state_dict(), path)


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def load_model(path, device):
    model = Cifar100_nn(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
