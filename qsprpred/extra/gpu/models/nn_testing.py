
from mols2grid import display
from torch.nn.functional import linear

# Přidání cesty k adresáři, kde se nachází tvůj skript
# sys.path.append(os.path.abspath('/Users/krynekt/DataspellProjects/Bakalarka/QSPRpred/qsprpred/extra/gpu/models'))
import sys, os
sys.path.insert(0, os.path.abspath('/Users/krynekt/DataspellProjects/Bakalarka/QSPRpred/'))
# Nyní můžeš naimportovat svůj skript
print(sys.path)
import pandas as pd

df = pd.read_csv('../../../../tutorials/tutorial_data/A2A_LIGANDS.tsv', sep='\t')

df.head()
from qsprpred.data import QSPRDataset

import os

os.makedirs("tutorial_output/data", exist_ok=True)

dataset = QSPRDataset(
    df=df,
    store_dir="tutorial_output/data",
    name="QuickStartDataset",
    target_props=[{"name": "pchembl_value_Mean", "task": "REGRESSION"}],
    random_state=42,
    overwrite=True
)

dataset.getDF()
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.data import RandomSplit

# Specifiy random split for creating the train (80%) and test set (20%)
rand_split = RandomSplit(test_fraction=0.2, dataset=dataset)

# calculate compound features and split dataset into train and test
dataset.prepareDataset(
    split=rand_split,
    feature_calculators=[MorganFP(radius=3, nBits=2048)],
)

print(f"Number of samples in train set: {len(dataset.y)}")
print(f"Number of samples in test set: {len(dataset.y_ind)}")

dataset.save()

import torch

from torch.nn import functional as f
from sklearn.metrics import mean_squared_error
from qsprpred.extra.gpu.models.neural_network import STFullyConnected


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = STFullyConnected(
        n_dim=dataset.X.shape[1],  # počet vstupních neuronů (počet deskriptorů)
        n_class=1,  # regresní úloha (1 výstup)
        gpus=[],
        device=device,
        batch_size=256,
        patience=20,
        tol=1e-4,
        is_reg=True,
        extra_layer=False,
        act_fun=f.selu,
    )
x.fit(dataset.X, dataset.y)
print(mean_squared_error(dataset.y_ind, x.predict(dataset.X_ind)))