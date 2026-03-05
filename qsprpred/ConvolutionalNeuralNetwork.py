import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import random
import copy
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from math import sqrt

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer=[32, 64, 128], kernel_size=3, padding=1, device="cpu", activation_func=nn.ReLU(), n_epochs=1000, learning_rate=1e-3, patience=10, dropout_frac = 0.5, batch_size=128, tol = 0, weight_decay=0, optimizer=optim.AdamW):
        """
        Inicializuje konvoluční neuronovou síť (CNN).

        Args:
            input_size (int): Celkový počet pixelů vstupního obrázku (příznaků).
            output_size (int): Počet výstupních tříd.
            layer (list, optional): Seznam celých čísel udávajících počet výstupních kanálů pro každou konvoluční vrstvu.
                                     Výchozí je [32, 64, 128].
            kernel_size (int, optional): Velikost konvolučního jádra. Výchozí je 3.
            padding (int, optional): Velikost paddingu pro konvoluční vrstvy. Výchozí je 1.
            device (str, optional): Zařízení, na kterém se bude model trénovat ('cpu' nebo 'cuda'). Výchozí je "cpu".
            activation_func (nn.Module, optional): Aktivační funkce použitá v síti. Výchozí je nn.ReLU().
            n_epochs (int, optional): Maximální počet epoch pro trénování. Výchozí je 1000.
            learning_rate (float, optional): Rychlost učení pro optimalizátor. Výchozí je 1e-3.
            patience (int, optional): Počet epoch bez zlepšení na validační sadě, po kterých se trénování zastaví.
                                      Výchozí je 10.
            dropout_frac (float, optional): Pravděpodobnost vypadnutí neuronu v dropout vrstvách (2D i standardní).
                                            Výchozí je 0.5.
            batch_size (int, optional): Velikost dávky pro trénování a validaci. Výchozí je 128.
            tol (float, optional): Tolerance pro zlepšení přesnosti při early stopping. Zlepšení menší než tato
                                   hodnota se nepočítá. Výchozí je 0.
            weight_decay (float, optional): L2 regularizace pro optimalizátor. Výchozí je 0.
            optimizer (torch.optim.Optimizer, optional): Optimalizační algoritmus. Výchozí je optim.AdamW.
        """
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.device = device
        self.set_seed(69)
        self.n_epochs = n_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.flatten = nn.Flatten()
        self.tol = tol
        self.layers = [
            nn.Conv2d(1, layer[0], kernel_size=kernel_size,padding=padding),
            nn.BatchNorm2d(layer[0]),
            activation_func,
            nn.Dropout2d(p=dropout_frac),
            nn.MaxPool2d(2, 2)
        ]
        w = sqrt(input_size)
        print(w)
        for i in range(1, len(layer)):
            self.layers.append(nn.Conv2d(layer[i-1], out_channels=layer[i], kernel_size=kernel_size,padding=padding))
            self.layers.append(nn.BatchNorm2d(layer[i]))
            self.layers.append(activation_func)
            self.layers.append(nn.Dropout2d(p=dropout_frac)),
            self.layers.append(nn.MaxPool2d(2, 2))
            w = w // 2
            print(w)
        w = int(w //2)
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(layer[-1]*w*w, 256))
        self.layers.append(activation_func)
        self.layers.append(nn.Dropout(p=dropout_frac))
        self.layers.append(nn.Linear(256, output_size))
        self.model = nn.Sequential(*self.layers)
        print(self.model)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.to(self.device)
        self.best_model_weights = None
        self.best_epoch = -1
        self.best_acc = -1
        self.val_acc_history = []


    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Trénuje konvoluční neuronovou síť na zadaných trénovacích datech.

        Implementuje early stopping na základě výkonu na validačních datech, pokud jsou poskytnuta.
        Na konci trénování načte váhy modelu s nejlepší dosaženou přesností na validační sadě.
        Data jsou zpracovávána v dávkách pomocí DataLoaderu.

        Args:
            X_train (pd.DataFrame): Trénovací data. Očekává se, že každý řádek je zploštělý obrázek.
            y_train (pd.Series): Trénovací data.
            X_valid (pd.DataFrame, optional): Validační data. 
            y_valid (pd.Series, optional): Validační data. 
        """
        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    
        if y_valid is not None and X_valid is not None:
            y_valid = torch.tensor(y_valid.to_numpy(), dtype=torch.long).to(self.device)
            X_valid = torch.tensor(X_valid.to_numpy(), dtype=torch.float32).to(self.device)
            print(X_valid.shape)
            X_valid = X_valid.view(-1, 1, 32, 32)
        X_train = X_train.view(-1, 1, 32, 32)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        last_improvement = 0
        for i in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
    
                output = self.forward(batch_X)
                loss = self.loss(output, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if X_valid is not None and y_valid is not None:
                self.model.eval()
                output_val_all = []
                with torch.no_grad():
                    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=self.batch_size)  # menší batch size
                    for batch_X, batch_y in valid_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        output_val = self.predict(batch_X)
                        output_val_all.append(output_val)
                    
                output_val_all = np.concatenate(output_val_all)  # Spoj všechny dávky do jednoho pole
                acc_val = accuracy_score(y_valid.cpu(), output_val_all)
                self.val_acc_history.append(acc_val)
                print(f"Epoch: {i}| Validation accuracy score = {acc_val:.4f} | Current max: {self.best_acc:.4f}")
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
    
        if self.best_model_weights is not None:
            self.load_state_dict(self.best_model_weights)
            print(f"Restored the best model at epoch {self.best_epoch} with validation accuracy {self.best_acc:.4f}")
        else:
            print("No best model weights were found.")
        print(f"Finished Training | Best validation accuracy = {self.best_acc:.4f} at epoch {self.best_epoch}")
    

    def forward(self, x):
        """
        Definuje dopředný průchod konvoluční neuronovou sítí. 

        Args:
            x (torch.Tensor): Vstupní tenzor s tvarem (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Výstupní tenzor (logits) s tvarem (batch_size, output_size).
        """
        return self.model(x)

    def predict(self, X, batch_size=256):
        """
        Provádí predikce na vstupních datech pomocí natrénovaného modelu.

        Vstupní data mohou být pandas DataFrame, NumPy pole nebo PyTorch tenzor.
        Data jsou automaticky převedena na PyTorch tenzora přesunuta na správné zařízení.
        Predikce se provádí v dávkách pro efektivní využití paměti.

        Args:
            X (pd.DataFrame or np.ndarray or torch.Tensor): Vstupní data pro predikci.
            batch_size_pred (int, optional): Velikost dávky pro predikci. Pokud None, použije se self.batch_size.
                                            Výchozí je None.

        Returns:
            np.ndarray: Pole predikovaných tříd (indexy tříd s nejvyšší pravděpodobností).
        """
        if not isinstance(X, torch.Tensor):
            if isinstance(X, pd.DataFrame):
                X = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            elif isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32).to(self.device)
            X= X.view(-1, 1, 32, 32)
        else:
            X = X.to(self.device)
        self.model.eval()  # přepnutí do eval režimu
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        preds = []
        with torch.no_grad():
            for batch_X, in loader:  # rozbalení jedno-prvkového tuple
                batch_X = batch_X.to(self.device)
                output = self.forward(batch_X)
                batch_preds = torch.argmax(output, dim=1)
                preds.append(batch_preds.cpu())
        
        return torch.cat(preds).numpy()



    def set_seed(self, seed=42):
        """
        Nastaví seed pro generátory náhodných čísel v PyTorch, NumPy a Pythonu
        pro zajištění reprodukovatelnosti experimentů.

        Args:
            seed (int, optional): Hodnota seedu. Výchozí je 42.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
