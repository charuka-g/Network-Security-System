import os
import sys
import pickle
import numpy as np
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from src.exception import NetworkSecurityException
from src.logger import logging
from src.config import ClassificationMetric


def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def write_yaml(file_path: str, content: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_object(file_path: str) -> object:
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            np.save(f, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def load_numpy_array(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "rb") as f:
            return np.load(f)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def get_classification_score(y_true, y_pred) -> ClassificationMetric:
    try:
        return ClassificationMetric(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred),
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    try:
        report = {}
        for name, model in models.items():
            gs = GridSearchCV(model, params[name], cv=3, scoring="f1")
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            report[name] = f1_score(y_test, model.predict(X_test))
            logging.info(f"  {name}: F1={report[name]:.4f}")
        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


class NetworkModel:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, x):
        try:
            return self.model.predict(self.preprocessor.transform(x))
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
