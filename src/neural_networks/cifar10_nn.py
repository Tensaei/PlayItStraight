import torch
import torch.nn as nn
import torch.nn.functional as F


class Cifar10_nn(nn.Module):

    def __init__(self, device):
        super(Cifar10_nn, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, stride=1, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1)

        self.fc1 = nn.Linear(in_features=6 * 6 * 256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=10)

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(-1, 6 * 6 * 256)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        logits = self.fc4(x)
        return logits#, F.softmax(logits)

    def detailed_forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(-1, 6 * 6 * 256)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        logits = self.fc4(x)
        return logits#, F.softmax(logits)

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


def load_model(path, device):
    model = Cifar10_nn(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
