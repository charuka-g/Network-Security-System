import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from src.exception import NetworkSecurityException
from src.logger import logging
from src.config import (
    DataValidationArtifact, DataTransformationConfig, DataTransformationArtifact,
    TARGET_COLUMN, IMPUTER_PARAMS,
)
from src.utils import save_numpy_array, save_object


class DataTransformation:
    def __init__(self, validation_artifact: DataValidationArtifact, config: DataTransformationConfig):
        self.validation_artifact = validation_artifact
        self.config = config

    def _get_preprocessor(self) -> Pipeline:
        return Pipeline([("imputer", KNNImputer(**IMPUTER_PARAMS))])

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.validation_artifact.valid_test_file_path)

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN].replace(-1, 0)
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN].replace(-1, 0)

            preprocessor = self._get_preprocessor()
            preprocessor.fit(X_train)

            X_train_t = preprocessor.transform(X_train)
            X_test_t = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_t, np.array(y_train)]
            test_arr = np.c_[X_test_t, np.array(y_test)]

            save_numpy_array(self.config.transformed_train_file_path, train_arr)
            save_numpy_array(self.config.transformed_test_file_path, test_arr)
            save_object(self.config.preprocessor_file_path, preprocessor)
            save_object("models/preprocessor.pkl", preprocessor)

            logging.info(f"Transformation done. Train: {train_arr.shape}, Test: {test_arr.shape}")
            return DataTransformationArtifact(
                preprocessor_file_path=self.config.preprocessor_file_path,
                transformed_train_file_path=self.config.transformed_train_file_path,
                transformed_test_file_path=self.config.transformed_test_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
