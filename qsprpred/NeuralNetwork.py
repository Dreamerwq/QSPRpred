import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, device="cpu", n_epochs=1000, learning_rate=1e-3, patience = 10):
        super(NeuralNetwork,self).__init__()
        self.device = device
        self.n_epochs = n_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.flatten = nn.Flatten()
        hidden_layers.append(output_size)
        hidden_layers.insert(0, input_size)
        layers = []
        for i in range(len(hidden_layers) - 2):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-2], hidden_layers[-1]))

        self.model = nn.Sequential(*layers)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        print(self.model)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        y_valid = torch.tensor(y_valid.to_numpy(), dtype=torch.long)
        best_acc = -1
        best_epoch = 0
        last_improvement = 0
        for i in range(self.n_epochs):
            output = self.forward(X_train)
            loss = self.loss(output, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if X_valid is not None and y_valid is not None:
                output_val = self.predict(X_valid)
                acc_val = accuracy_score(y_valid, output_val)
                print(f"Epoch: {i}| accuracy score = {acc_val}")
                if acc_val > best_acc:
                    last_improvement = 0
                    best_acc = acc_val
                    best_epoch = i
                else:
                    if last_improvement >= self.patience:
                        print(f"Early stopping at epoch {i-self.patience}")
                        break
                    else:
                        last_improvement += 1
            if i % 100 == 0:
                print(f"Epoch {i}: Loss = {loss.item()}")
        print(f"Finished Training | Validation accuracy score = {best_acc}")


    def forward(self, x):
        return self.model(x)

    def predict(self, X):
        X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        outputs = self.forward(X)
        return torch.argmax(outputs, dim=1)

