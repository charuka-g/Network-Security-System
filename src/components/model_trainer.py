import os
import sys

import mlflow
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.exception import NetworkSecurityException
from src.logger import logging
from src.config import DataTransformationArtifact, ModelTrainerConfig, ModelTrainerArtifact
from src.utils import (
    evaluate_models,
    get_classification_score,
    load_numpy_array,
    load_object,
    save_object,
    NetworkModel,
)
from src.integrations.dagshub_mlflow import init_dagshub


class ModelTrainer:
    def __init__(self, transformation_artifact: DataTransformationArtifact, config: ModelTrainerConfig):
        self.transformation_artifact = transformation_artifact
        self.config = config

    def _setup_mlflow(self):
        init_dagshub()

    def _track_mlflow(self, model, metric):
        try:
            with mlflow.start_run():
                mlflow.log_metric("f1_score", metric.f1_score)
                mlflow.log_metric("precision", metric.precision_score)
                mlflow.log_metric("recall", metric.recall_score)
                mlflow.sklearn.log_model(model, "model")
        except Exception:
            logging.warning("MLflow tracking skipped (URI not configured or unreachable)")

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            self._setup_mlflow()

            train_arr = load_numpy_array(self.transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array(self.transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "Random Forest": RandomForestClassifier(verbose=0),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=0),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "AdaBoost": AdaBoostClassifier(),
            }
            params = {
                "Random Forest": {"n_estimators": [32, 128, 256]},
                "Decision Tree": {"criterion": ["gini", "entropy"]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [64, 128],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "n_estimators": [64, 128],
                    "learning_rate": [0.1, 0.01],
                },
            }

            logging.info("Starting model evaluation with GridSearchCV...")
            report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(report, key=report.get)
            best_score = report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} (Test F1={best_score:.4f})")

            if best_score < self.config.expected_accuracy:
                raise Exception(
                    f"No model met the expected accuracy of {self.config.expected_accuracy}. "
                    f"Best was {best_model_name} with F1={best_score:.4f}"
                )

            train_metric = get_classification_score(y_train, best_model.predict(X_train))
            test_metric = get_classification_score(y_test, best_model.predict(X_test))

            self._track_mlflow(best_model, test_metric)

            preprocessor = load_object(self.transformation_artifact.preprocessor_file_path)
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            save_object(self.config.trained_model_file_path, network_model)
            save_object("models/model.pkl", best_model)

            logging.info(
                f"Model saved. Train F1={train_metric.f1_score:.4f}, Test F1={test_metric.f1_score:.4f}"
            )
            return ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_file_path,
                train_metric=train_metric,
                test_metric=test_metric,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
