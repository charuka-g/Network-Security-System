import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from src.exception import NetworkSecurityException
from src.logger import logging
from src.config import (
    DataIngestionArtifact, DataValidationConfig, DataValidationArtifact,
    SCHEMA_FILE_PATH,
)
from src.utils import read_yaml, write_yaml


class DataValidation:
    def __init__(self, ingestion_artifact: DataIngestionArtifact, config: DataValidationConfig):
        try:
            self.ingestion_artifact = ingestion_artifact
            self.config = config
            self.schema = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _validate_columns(self, df: pd.DataFrame) -> bool:
        expected = set(self.schema["columns"])
        actual = set(df.columns)
        if expected == actual:
            return True
        logging.warning(f"Column mismatch — missing: {expected - actual}, extra: {actual - expected}")
        return False

    def _detect_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        report = {}
        drift_found = False
        for col in train_df.columns:
            stat = ks_2samp(train_df[col], test_df[col])
            drifted = stat.pvalue < 0.05
            if drifted:
                drift_found = True
            report[col] = {"p_value": float(stat.pvalue), "drift": drifted}
        write_yaml(self.config.drift_report_file_path, report)
        if drift_found:
            logging.warning("Data drift detected in some columns — check drift_report.yaml")
        return not drift_found

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)

            cols_valid = self._validate_columns(train_df) and self._validate_columns(test_df)
            no_drift = self._detect_drift(train_df, test_df)

            os.makedirs(os.path.dirname(self.config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.config.valid_train_file_path, index=False)
            test_df.to_csv(self.config.valid_test_file_path, index=False)

            logging.info(f"Validation status: columns_ok={cols_valid}, no_drift={no_drift}")
            return DataValidationArtifact(
                validation_status=cols_valid and no_drift,
                valid_train_file_path=self.config.valid_train_file_path,
                valid_test_file_path=self.config.valid_test_file_path,
                drift_report_file_path=self.config.drift_report_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
