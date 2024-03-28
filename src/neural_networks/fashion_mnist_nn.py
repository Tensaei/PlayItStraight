import torch
import torch.nn as nn
import torch.nn.functional as F


class Fashion_MNIST_nn(nn.Module):

    def __init__(self, device):
        super(Fashion_MNIST_nn, self).__init__()
        self.device = device
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        self.to(self.device)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return F.log_softmax(out)

    def detailed_forward(self, x):
        x = x.reshape((1, 1, 28, 28)).type(torch.float32)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x, F.log_softmax(x)

    def _train_epoch(self, optimizer, train_loader, criterion):
        self.train()
        for _, (data, target, _) in enumerate(train_loader):
            optimizer.zero_grad()
            output = self(data.to(self.device))
            loss = criterion(output.to(self.device), target.to(self.device))
            loss.backward(retain_graph=True)
            optimizer.step()

    def evaluate(self, criterion, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        with (torch.no_grad()):
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

    def predict(self, x):
        return self.forward(torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0).to(self.device)).cpu().squeeze().detach().numpy()

    def loss_gradient(self, x, y):
        prediction = self(x)
        error = prediction - y

        # Calcola la derivata della funzione di perdita del modello rispetto alla predizione del modello.
        loss_gradient_prediction = error * self.activation(prediction)

        # Calcola la derivata della funzione di attivazione dell'ultimo layer rispetto alla predizione del modello.
        activation_gradient = self.activation_gradient(prediction)

        # Calcola la derivata della funzione di perdita del modello rispetto ai parametri del modello.
        gradient = loss_gradient_prediction * activation_gradient

        return gradient

    def get_embedding_dim(self):
        return 10

    def update(self, path):
        pass


def load_model(path, device):
    model = MNIST_nn(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
