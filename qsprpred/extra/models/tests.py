"""
Test module for testing extra models.

"""

from typing import Type

from parameterized import parameterized
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor

from qsprpred.extra.data.descriptors.calculators import ProteinDescriptorCalculator
from qsprpred.extra.data.descriptors.sets import ProDec
from qsprpred.extra.data.tables.pcm import PCMDataSet
from qsprpred.extra.data.utils.msa_calculator import ClustalMSA
from qsprpred.tasks import TargetProperty, TargetTasks
from ..data.utils.testing.path_mixins import DataSetsMixInExtras
from ..models.pcm import SklearnPCMModel
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.check_mixins import ModelCheckMixIn
from ...utils.testing.path_mixins import ModelDataSetsPathMixIn
from ..models.random import RandomModel


class ModelDataSetsMixInExtras(ModelDataSetsPathMixIn, DataSetsMixInExtras):
    """This class holds the tests for testing models in extras."""


class TestPCM(ModelDataSetsMixInExtras, ModelCheckMixIn, QSPRTestCase):
    """Test class for testing PCM models."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def getModel(
        self,
        name: str,
        alg: Type | None = None,
        dataset: PCMDataSet | None = None,
        parameters: dict | None = None,
        random_state: int | None = None,
    ):
        """Initialize dataset and model.

        Args:
            name (str): Name of the model.
            alg (Type | None): Algorithm class.
            dataset (PCMDataSet | None): Dataset to use.
            parameters (dict | None): Parameters to use.
            random_state (int | None): Random seed to use.

        Returns:
            SklearnPCMModel: Initialized model.
        """
        return SklearnPCMModel(
            base_dir=self.generatedModelsPath,
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters,
            random_state=random_state,
        )

    @parameterized.expand(
        [
            (
                alg_name,
                [{"name": "pchembl_value_Median", "task": TargetTasks.REGRESSION}],
                alg_name,
                alg,
                random_state,
            )
            for alg, alg_name in ((XGBRegressor, "XGBR"),)
            for random_state in ([None], [1, 42], [42, 42])
        ]
        + [
            (
                alg_name,
                [{"name": "pchembl_value_Median", "task": TargetTasks.REGRESSION}],
                alg_name,
                alg,
                [None],
            )
            for alg, alg_name in (
                (PLSRegression, "PLSR"),
                (SVR, "SVR"),
            )
        ]
        + [
            (
                alg_name,
                [
                    {
                        "name": "pchembl_value_Median",
                        "task": TargetTasks.SINGLECLASS,
                        "th": [6.5],
                    }
                ],
                alg_name,
                alg,
                random_state,
            )
            for alg, alg_name in (
                (RandomForestClassifier, "RFC"),
                (XGBClassifier, "XGBC"),
            )
            for random_state in ([None], [1, 42], [42, 42])
        ]
    )
    def testRegressionBasicFitPCM(
        self,
        _,
        props: list[TargetProperty | dict],
        model_name: str,
        model_class: Type,
        random_state: list[int | None],
    ):
        """Test model training for regression models.

        Args:
            _: Name of the test.
            props (list[TargetProperty | dict]): List of target properties.
            model_name (str): Name of the model.
            model_class (Type): Class of the model.

        """
        if model_name not in ["SVR", "PLSR"]:
            parameters = {"n_jobs": self.nCPU}
        else:
            parameters = None
        # initialize dataset
        prep = self.getDefaultPrep()
        prep["feature_calculators"] = prep["feature_calculators"] + [
            ProteinDescriptorCalculator(
                desc_sets=[ProDec(sets=["Sneath"])],
                msa_provider=ClustalMSA(self.generatedDataPath),
            )
        ]
        dataset = self.createPCMDataSet(
            name=f"{model_name}_{props[0]['task']}_pcm",
            target_props=props,
            preparation_settings=prep,
            random_state=random_state[0],
        )
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_{props[0]['task']}",
            alg=model_class,
            dataset=dataset,
            parameters=parameters,
            random_state=random_state[0],
        )
        self.fitTest(model)
        predictor = SklearnPCMModel(
            name=f"{model_name}_{props[0]['task']}", base_dir=model.baseDir
        )
        pred_use_probas, pred_not_use_probas = self.predictorTest(
            predictor, protein_id=dataset.getDF()["accession"].iloc[0]
        )
        if random_state[0] is not None:
            model = self.getModel(
                name=f"{model_name}_{props[0]['task']}",
                alg=model_class,
                dataset=dataset,
                parameters=parameters,
                random_state=random_state[1],
            )
            self.fitTest(model)
            predictor = SklearnPCMModel(
                name=f"{model_name}_{props[0]['task']}", base_dir=model.baseDir
            )
            self.predictorTest(
                predictor,
                protein_id=dataset.getDF()["accession"].iloc[0],
                expect_equal_result=random_state[0] == random_state[1],
                expected_pred_use_probas=pred_use_probas,
                expected_pred_not_use_probas=pred_not_use_probas,
            )

class TestRandom(ModelDataSetsMixInExtras, ModelCheckMixIn, QSPRTestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def getModel(
        self,
        name: str,
        dataset: PCMDataSet | None = None,
        parameters: dict | None = None,
        random_state: int | None = None,
    ):
        """Initialize dataset and model.

        Args:
            name (str): Name of the model.
            alg (Type | None): Algorithm class.
            dataset (PCMDataSet | None): Dataset to use.
            parameters (dict | None): Parameters to use.
            random_state (int | None): Random seed to use.

        Returns:
            RandomModel: Initialized model.
        """
        return RandomModel(
            base_dir=self.generatedModelsPath,
            data=dataset,
            name=name,
            parameters=parameters,
            random_state=random_state,
        )

    @parameterized.expand(
        [
            ('RandomModel', random_state)
            for random_state in ([None], [1, 42], [42, 42])
        ]
    )
    def testRegressionMultiTaskFit(self, model_name, random_state: list[int | None]):
        """Test model training for multitask regression models."""
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[
                {
                    "name": "fu",
                    "task": TargetTasks.REGRESSION,
                    "imputer": SimpleImputer(strategy="mean"),
                },
                {
                    "name": "CL",
                    "task": TargetTasks.REGRESSION,
                    "imputer": SimpleImputer(strategy="mean"),
                },
            ],
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"RandomModel_multitask_regression",
            dataset=dataset,
            random_state=random_state[0],
        )
        self.fitTest(model)
        predictor = RandomModel(
            name=f"RandomModel_multitask_regression", base_dir=model.baseDir
        )
        pred_use_probas, pred_not_use_probas = self.predictorTest(predictor)
        if random_state[0] is not None:
            model = self.getModel(
                name=f"RandomModel_multitask_regression",
                dataset=dataset,
                random_state=random_state[1],
            )
            self.fitTest(model)
            predictor = RandomModel(
                name=f"RandomModel_multitask_regression", base_dir=model.baseDir
            )
            self.predictorTest(
                predictor,
                expect_equal_result=random_state[0] == random_state[1],
                expected_pred_use_probas=pred_use_probas,
                expected_pred_not_use_probas=pred_not_use_probas,
            )
