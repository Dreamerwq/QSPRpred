import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import random
import copy
import pandas as pd

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, device="cpu", n_epochs=1000, learning_rate=1e-3, patience = 10, optimizer=optim.AdamW, seed=69, tol=0, dropout_frac=0.25):
        """
        Inicializace neuronové sítě.

        Args:
            input_size (int): Počet příznaků.
            hidden_layers (list): List intů udávajících počet neuronů v každé skryté vrstvě.
            output_size (int): Počet predikovaných tříd.
            device (str, optional): Zařízení, na kterém se bude model trénovat ('cpu' nebo 'cuda'). Výchozí je "cpu".
            n_epochs (int, optional): Maximální počet epoch pro trénování. Výchozí je 1000.
            learning_rate (float, optional): Rychlost učení pro optimalizátor při trénování. Výchozí je 1e-3.
            patience (int, optional): Počet epoch bez zlepšení na validační sadě, po kterých se trénování zastaví (early stopping). Výchozí je 10.
            optimizer (torch.optim.Optimizer, optional): Optimalizační algoritmus. Výchozí je optim.AdamW.
            seed (int, optional): Seed pro reprodukovatelnost. Výchozí je 69.
            tol (float, optional): Tolerance pro zlepšení přesnosti při early stopping. Zlepšení menší než tato hodnota se nepočítá. Výchozí je 0.
            dropout_frac (float, optional): Pravděpodobnost vypadnutí neuronu v dropout vrstvě. Výchozí je 0.25.
        """
        super(NeuralNetwork,self).__init__()
        self.device = device
        self.set_seed(seed=seed)
        self.n_epochs = n_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.flatten = nn.Flatten()
        self.dropout_frac=dropout_frac
        self.tol= tol
        hidden_layers.append(output_size)
        hidden_layers.insert(0, input_size)
        layers = []
        for i in range(len(hidden_layers) - 2):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Dropout(p=self.dropout_frac))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-2], hidden_layers[-1]))

        self.model = nn.Sequential(*layers)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.to(self.device)
        self.best_model_weights = None
        self.best_epoch = -1
        self.best_acc = -1

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Funkce trénuje neuronovou síť na zadaných trénovacích datech.

        Implementuje early stopping na základě výkonu na validačních datech.
        Na konci trénování načte váhy modelu s nejlepší dosaženou přesností na validační sadě.

        Args:
            X_train (pd.DataFrame): Trénovací data.
            y_train (pd.Series): Trénovací data.
            X_valid (pd.DataFrame, optional): Validační data. 
            y_valid (pd.Series, optional): Validační data. 
        """
        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long).to(self.device).squeeze()
        if X_valid is not None and y_valid is not None:
            X_valid = torch.tensor(X_valid.to_numpy(), dtype=torch.float32).to(self.device)
            y_valid = torch.tensor(y_valid.to_numpy(), dtype=torch.long).to(self.device).squeeze()
        last_improvement = 0
        for i in range(self.n_epochs):
            # Forward pass
            self.train() 
            output = self.forward(X_train)
            loss = self.loss(output, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.eval() 
            if X_valid is not None and y_valid is not None:
                output_val = self.predict(X_valid)
                acc_val = accuracy_score(y_valid.cpu(), output_val) 
                print(f"Epoch: {i}| Validation accuracy score = {acc_val:.4f}")
                if acc_val > self.best_acc + self.tol:
                    last_improvement = 0
                    self.best_acc = acc_val
                    self.best_epoch = i
                    self.best_model_weights = copy.deepcopy(self.state_dict())
                else:
                    if last_improvement >= self.patience:
                        print(f"Early stopping at epoch {i-self.patience}")
                        break
                    else:
                        last_improvement += 1
                        
        self.load_state_dict(self.best_model_weights)




    def set_seed(self, seed):
        """
        Nastaví seed pro generátory náhodných čísel v PyTorch a NumPy pro zajištění reprodukovatelnosti.

        Args:
            seed (int): Hodnota seedu.
        """
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    def forward(self, x):
        """
        Definuje dopředný průchod (forward pass) neuronovou sítí. 

        Args:
            x (torch.Tensor): Vstupní tenzor.

        Returns:
            torch.Tensor: Výstupní tenzor.
        """
        return self.model(x)

    def predict(self, X):
        """
        Provádí predikce na vstupních datech.

        Data jsou automaticky převedena na PyTorch tenzor.

        Args:
            X (pd.DataFrame or np.ndarray or torch.Tensor): Vstupní data pro predikci.

        Returns:
            np.ndarray: Pole predikovaných tříd (indexy tříd s nejvyšší pravděpodobností).
        """
        if not isinstance(X, torch.Tensor):
            if isinstance(X, pd.DataFrame):
                X = torch.tensor(X.values, dtype=torch.float32)
            elif isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        outputs = self.forward(X)
        predictions = torch.argmax(outputs, dim=1)
        return predictions.cpu().numpy()  # Přesun na CPU a převod na numpy

