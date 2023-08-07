"""
tests

Created by: Martin Sicho
On: 12.05.23, 18:31
"""
import os
from typing import Type
from unittest import TestCase

import torch
from parameterized import parameterized

from ...data.data import QSPRDataset
from ...data.utils.datasplitters import RandomSplit
from ...data.utils.descriptorcalculator import CustomDescriptorsCalculator
from ...data.utils.descriptorsets import DataFrameDescriptorSet
from ...deep.models.models import QSPRDNN
from ...deep.models.chemprop import Chemprop
from ...deep.models.neural_network import STFullyConnected
from ...models.tasks import TargetTasks
from ...models.tests import ModelDataSetsMixIn, ModelTestMixIn

GPUS = list(range(torch.cuda.device_count()))


class NeuralNet(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the QSPRDNN class."""

    qsprModelsPath = f"{os.path.dirname(__file__)}/test_files/qspr/models"

    @property
    def gridFile(self):
        """Return the path to the grid file with test
        search spaces for hyperparameter optimization.
        """
        return f"{os.path.dirname(__file__)}/test_files/search_space_test.json"

    @staticmethod
    def getModel(
        base_dir: str,
        name: str,
        alg: Type | None = None,
        dataset: QSPRDataset = None,
        parameters: dict | None = None,
    ):
        """Initialize model with data set.

        Args:
            base_dir: Base directory for model.
            name: Name of the model.
            alg: Algorithm to use.
            dataset: Data set to use.
            parameters: Parameters to use.
        """
        return QSPRDNN(
            base_dir=base_dir,
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters,
            gpus=GPUS,
            patience=3,
            tol=0.02,
        )

    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, alg_name, alg, th)
            for alg, alg_name, task, th in (
                (STFullyConnected, "STFullyConnected", TargetTasks.REGRESSION, None),
                (STFullyConnected, "STFullyConnected", TargetTasks.SINGLECLASS, [6.5]),
                (
                    STFullyConnected,
                    "STFullyConnected",
                    TargetTasks.MULTICLASS,
                    [0, 1, 10, 1100],
                ),
            )
        ]
    )
    def test_qsprpred_model(
        self, _, task: TargetTasks, alg_name: str, alg: Type, th: float
    ):
        """Test the QSPRDNN model in one configuration.

        Args:
            task: Task to test.
            alg_name: Name of the algorithm.
            alg: Algorithm to use.
            th: Threshold to use for classification models.
        """
        # initialize dataset
        dataset = self.create_large_dataset(
            name=f"{alg_name}_{task}",
            target_props=[{
                "name": "CL",
                "task": task,
                "th": th
            }],
            preparation_settings=self.get_default_prep(),
        )

        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.getModel(
            base_dir=self.qsprModelsPath, name=f"{alg_name}", alg=alg, dataset=dataset
        )
        self.fitTest(model)
        predictor = QSPRDNN(name=alg_name, base_dir=model.baseDir)
        self.predictorTest(predictor)


class ChemProp(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the QSPRDNN class."""

    qsprModelsPath = f"{os.path.dirname(__file__)}/test_files/qspr/models"

    @property
    def gridFile(self):
        """Return the path to the grid file with test
        search spaces for hyperparameter optimization.
        """
        return f"{os.path.dirname(__file__)}/test_files/search_space_test.json"

    @staticmethod
    def getModel(
        base_dir: str,
        name: str,
        dataset: QSPRDataset = None,
        parameters: dict | None = None,
    ):
        """Initialize model with data set.

        Args:
            base_dir: Base directory for model.
            name: Name of the model.
            alg: Algorithm to use.
            dataset: Data set to use.
            parameters: Parameters to use.
        """
        return Chemprop(
            base_dir=base_dir, data=dataset, name=name, parameters=parameters
        )

    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, alg_name, th) for alg_name, task, th in (
                ("MoleculeModel", TargetTasks.REGRESSION, None),
                ("MoleculeModel", TargetTasks.SINGLECLASS, [6.5]),
                (
                    "MoleculeModel",
                    TargetTasks.MULTICLASS,
                    [0, 1, 10, 1100],
                ),
            )
        ]
    )
    def test_qsprpred_model(self, _, task: TargetTasks, alg_name: str, th: float):
        """Test the QSPRDNN model in one configuration.

        Args:
            task: Task to test.
            alg_name: Name of the algorithm.
            alg: Algorithm to use.
            th: Threshold to use for classification models.
        """
        # initialize dataset
        dataset = self.create_large_dataset(
            name=f"{alg_name}_{task}",
            target_props=[{
                "name": "CL",
                "task": task,
                "th": th
            }],
            preparation_settings=None
        )
        dataset.prepareDataset(split=RandomSplit(dataset=dataset, test_fraction=0.2))
        # Set SMILES as custom descriptor
        feature_calculator = CustomDescriptorsCalculator(
            desc_sets=[
                DataFrameDescriptorSet(dataset.getDF()[["SMILES"]]),
            ]
        )
        dataset.addCustomDescriptors(feature_calculator, dtype=str)
        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.getModel(
            base_dir=self.qsprModelsPath, name=f"{alg_name}", dataset=dataset
        )
        self.fitTest(model)
        predictor = Chemprop(name=alg_name, base_dir=model.baseDir)
        self.predictorTest(predictor)
