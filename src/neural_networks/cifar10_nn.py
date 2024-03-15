import torch
import torch.nn as nn
import torch.nn.functional as F


class Cifar10_nn(nn.Module):

    def __init__(self, device):
        super(Cifar10_nn, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.to(self.device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

    def save(self, path):
        torch.save(self.state_dict(), path)


def load_model(path, device):
    model = Cifar10_nn(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
