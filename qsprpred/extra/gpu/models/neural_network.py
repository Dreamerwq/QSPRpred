"""
This module holds the base class for DNN models
as well as fully connected NN subclass.
"""
print("lol")
import inspect
from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as f
from torch.utils.data import DataLoader, TensorDataset

#from .base_torch import QSPRModelPyTorchGPU, DEFAULT_TORCH_GPUS
from qsprpred.logs import logger
from ....models.monitors import BaseMonitor, FitMonitor
from torch.optim.lr_scheduler import *
from sklearn.metrics import matthews_corrcoef
import copy


class Base(nn.Module):
    """Base structure for all classification/regression DNN models.

    Mainly, it provides the general methods for training, evaluating model and
    predicting the given data.

    Attributes:
        device (torch.device):
            device to run the model on
        n_epochs (int):
            (maximum) number of epochs to train the model
        lr (float):
            learning rate
        batch_size (int):
            batch size
        patience (int):
            number of epochs to wait before early stop if no progress on validation
            set score, if patience = -1, always train to `n_epochs`
        tol (float):
            minimum absolute improvement of metric necessary to count as progress
            on best validation score
        seed (int):
            
    """

    def __init__(
            self,
            device: str = "cpu",
            n_epochs: int = 1000,
            lr: float = 1e-4,
            batch_size: int = 256,
            patience: int = 50,
            tol: float = 0,
            weight_decay: float = 1e-4,
            optimizer = optim.AdamW,
            seed=69,
            print_outputs = 0 # 0: No output, >0: print final output, >1: print each epoch output
    ):
        """
        Initialize the DNN model with training configuration.
        Args:
            device (str): 
                Device to run the model on ('cpu', 'cuda',...).
            n_epochs (int): 
                Maximum number of training epochs.
            lr (float): 
                Learning rate for the optimizer.
            batch_size (int): 
                Number of samples per training batch.
            patience (int): 
                Number of epochs to wait for improvement on validation loss before early stopping.
                If set to -1, training continues for all `n_epochs` regardless of validation performance.
            tol (float): 
                Minimum improvement in validation loss to be considered as progress.
            weight_decay (float): 
                Weight decay (L2 penalty) for the optimizer.
            optimizer (torch.optim.Optimizer): 
                Optimizer class to use for training (default: 'torch.optim.AdamW').
            seed (int): 
                Random seed for reproducibility.
            print_outputs (int): 
                Verbosity level for training outputs.
                0 = No output,
                >0 = Output after final epoch,
                >1 = Output after every epoch.
        """
        super().__init__()
        self.seed = seed
        self.set_seed(seed=seed)
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.tol = tol
        self.device = torch.device(device)
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.print_outputs = print_outputs



    def fit(
            self,
            X_train,
            y_train,
            X_valid=None,
            y_valid=None,
            monitor: FitMonitor | None = None,
    ) -> int:
        """
        Train the model on the provided training data with optional validation and early stopping.
        Args:
            X_train (pd.DataFrame or torch.Tensor): 
                Training features.
            y_train (pd.Series or torch.Tensor): 
                Training labels.
            X_valid (pd.DataFrame or torch.Tensor, optional): 
                Validation features for monitoring performance (default: None).
            y_valid (pd.Series or torch.Tensor, optional): 
                Validation labels (default: None).
            monitor (FitMonitor, optional): 
                Custom training monitor for logging and callbacks (default: BaseMonitor()).
        Returns:
            tuple:
                - self: the trained model.
                - last_save (int): the epoch index of the best model (based on validation loss).
        """
        self.to(self.device)
    
        monitor = BaseMonitor() if monitor is None else monitor


        train_loader = self.getDataLoader(X_train, y_train)
        valid_loader = self.getDataLoader(X_valid, y_valid) if X_valid is not None and y_valid is not None else None
        
        patience = self.patience if valid_loader is not None else -1
        optimizer = self.optim if "optim" in self.__dict__ else self.optimizer(self.parameters(), lr=self.lr)
    
        # Weighted loss for imbalance
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        pos_weight_val = (y_tensor == 0).sum() / (y_tensor == 1).sum()
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
        best_loss = np.inf
        best_weights = copy.deepcopy(self.state_dict())
        last_save = 0
    
        # Scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr * 10,
            total_steps=self.n_epochs * len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
            final_div_factor=1e4,
            div_factor=25.0,
        )
    
        for epoch in range(self.n_epochs):
            monitor.onEpochStart(epoch)
            loss = None
    
            self.train()
            for i, (Xb, yb) in enumerate(train_loader):
                monitor.onBatchStart(i)
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                
                y_ = self(Xb, is_train=True)
                # Remove potential NaNs
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
    
                loss = self.criterion(y_, yb)
                loss.backward()
                optimizer.step()
                scheduler.step()
                monitor.onBatchEnd(i, float(loss))
    
            if valid_loader is not None:
                loss_valid = self.evaluate(valid_loader)
                pred = self.predict(X_valid) > 0.5
                if self.print_outputs > 1:
                    print(f"Epoch {epoch + 1} | Train Loss: {loss.item():.4f} | Valid Loss: {loss_valid:.4f} | MCC: {matthews_corrcoef(pred, y_valid):.4f}")
                if loss_valid + self.tol < best_loss:
                    best_loss = loss_valid
                    best_weights = copy.deepcopy(self.state_dict())
                    last_save = epoch
                elif epoch - last_save > patience:
                    if self.print_outputs > 0:
                        print(f"Early stopping at epoch {epoch + 1} | Best Valid Loss: {best_loss:.4f}")
                    break
    
                monitor.onEpochEnd(epoch, loss.item(), loss_valid)
            else:
                monitor.onEpochEnd(epoch, loss.item())
    
        self.load_state_dict(best_weights)
        return self, last_save





    def evaluate(self, loader) -> float:
        """Evaluate the performance of the DNN model.

        Args:
            loader (torch.util.data.DataLoader):
                data loader for test set,
                including m X n target FloatTensor and l X n label FloatTensor
                (m is the No. of sample, n is the No. of features, l is the
                No. of classes or tasks)

        Return:
            loss (float):
                the average loss value based on the calculation of loss
                function with given test set.
        """
        self.to(self.device)
        self.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                y_ = self.forward(Xb)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                
                batch_size = yb.size(0)
                loss = self.criterion(y_, yb)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        return total_loss / total_samples if total_samples > 0 else float("inf")
        

    def predict(self, X_test) -> np.ndarray:
        """Predict the probability of each sample in the dataset."""
        self.to(self.device)
        self.eval()
        # If already a DataLoader, skip wrapping
        if isinstance(X_test, DataLoader):
            loader = X_test
        else:
            loader = self.getDataLoader(X_test)
    
        score = []
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                X_b = batch[0]  # In case it's (X, y)
            else:
                X_b = batch  # In case it's just X
    
            X_b = X_b.to(self.device)
            y_ = self.forward(X_b)
            score.append(y_.detach().cpu())
        return torch.sigmoid(torch.cat(score, dim=0)).numpy()


    @classmethod
    def _get_param_names(cls) -> list:
        """Get the class parameter names.

        Function copied from sklearn.base_estimator!

        Returns:
            parameter names (list): list of the class parameter names.
        """
        init_signature = inspect.signature(cls.__init__)
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True) -> dict:
        """Get parameters for this estimator.

        Function copied from sklearn.base_estimator!

        Args:
            deep (bool): If True, will return the parameters for this estimator

        Returns:
            params (dict): Parameter names mapped to their values.
        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params) -> "Base":
        """Set the parameters of this estimator.

        Function copied from sklearn.base_estimator!
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Args:
            **params : dict Estimator parameters.

        Returns:
            self : estimator instance
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        # grouped by prefix
        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        return self

    def getDataLoader(self, X, y=None):
        """Convert data to tensors and get generator over dataset with dataloader.

        Args:
            X (numpy 2d array): input dataset
            y (numpy 1d column vector): output data
        """
        # if pandas dataframe is provided, convert it to numpy array
        if hasattr(X, "values"):
            X = X.values
        if y is not None and hasattr(y, "values"):
            y = y.values
        if y is None:
            tensordataset = torch.Tensor(X)
            return DataLoader(
                tensordataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            tensordataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
            # Create a generator seeded to your trial or global seed
            g = torch.Generator()
            g.manual_seed(self.seed)
            return DataLoader(
                tensordataset,
                batch_size=self.batch_size,
                shuffle=True,
                generator=g,
                num_workers=0  # keep at 0 for full reproducibility
            )

    @staticmethod
    def set_seed(seed):
        print(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


class STFullyConnected(Base):
    """Single task DNN classification/regression model.

    It contains four fully connected layers between which are
    dropout layers for robustness.

    Attributes:
        n_dim (int): the No. of columns (features) for input tensor
        n_class (int): the No. of columns (classes) for output tensor.
        device (torch.cude): device to run the model on
        gpus (list): list of gpu ids to run the model on
        n_epochs (int): max number of epochs
        lr (float): neural net learning rate
        batch_size (int): batch size for training
        patience (int): early stopping patience
        tol (float): early stopping tolerance
        dropout_frac (float): dropout fraction
        criterion (torch.nn.Module): the loss function
        dropout (torch.nn.Module): the dropout layer
        neuron_layers (list(nn.Linear)): List of neuron layers
        act_fun (torch.nn.Functional): the activation function
        final_layer_activation (torch.nn.Module): the activation function
    """

    def __init__(
            self,
            n_dim,
            n_class,
            device,
            act_fun=f.relu,
            n_epochs=100,
            lr=1e-4 ,
            batch_size=256,
            patience=50,
            tol=0,
            neuron_layers=[2048, 1024],
            dropout_frac=0.25,
            weight_decay=0,
            optimizer=optim.AdamW,
            seed = 42,
            print_outputs = 0
    ):
        """Initialize the STFullyConnected model.

        Args:
            n_dim (int):
                the No. of columns (features) for input tensor
            n_class (int):
                the No. of columns (classes) for output tensor.
            device (torch.cude):
                device to run the model on
            gpus (list):
                list of gpu ids to run the model on
            n_epochs (int):
                max number of epochs
            lr (float):
                neural net learning rate
            batch_size (int):
                batch size
            patience (int):
                number of epochs to wait before early stop if no progress on
                validation set score, if patience = -1, always train to n_epochs
            tol (float):
                minimum absolute improvement of loss necessary to
                count as progress on best validation score
            neurons_h1 (int):
                number of neurons in first hidden layer
            neurons_hx (int):
                number of neurons in other hidden layers
            extra_layer (bool):
                add third hidden layer
            dropout_frac (float):
                dropout fraction
        """
        super().__init__(
            device=device,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            tol=tol,
            weight_decay=weight_decay,
            optimizer=optimizer,
            seed=seed,
             print_outputs=print_outputs
        )
        self.n_dim = n_dim
        self.dropout_frac = dropout_frac
        self.dropout = None
        self.neuron_layers = neuron_layers
        self.layers = []
        self.criterion = None
        self.act_fun = act_fun
        self.weight_decay = weight_decay
        self.initModel()

    def initModel(self):
        """Define the layers of the model."""
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.n_dim, self.neuron_layers[0]))
        for i in range(1, len(self.neuron_layers)):
            self.layers.append(nn.Linear(self.neuron_layers[i - 1], self.neuron_layers[i]))
        self.layers.append(nn.Linear(self.neuron_layers[-1], 1))
        self.dropout = nn.Dropout(self.dropout_frac)

    def set_params(self, **params) -> "STFullyConnected":
        """Set parameters and re-initialize model.

        Args:
            **params: parameters to be set

        Returns:
            self (STFullyConnected): the model itself
        """
        super().set_params(**params)
        self.initModel()
        return self

    def forward(self, X, is_train=False) -> torch.Tensor:
        """Invoke the class directly as a function.

        Args:
            X (FloatTensor):
                m X n FloatTensor, m is the No. of samples, n is
                the No. of features.
            is_train (bool, optional):
                is it invoked during training process (True) or
                just for prediction (False)
        Returns:
            y (FloatTensor): m X n FloatTensor, m is the No. of samples,
                n is the No. of classes
        """

        y = self.act_fun(self.layers[0](X))
        for i in range(1, len(self.layers) - 1):
            y = self.act_fun(self.layers[i](y))
            if is_train:
                y = self.dropout(y)  # Apply dropout only during training
        # SoftMax from BCEWithLogitsLoss
        y = self.layers[-1](y)
        return y