import os
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# ─── Pipeline Constants ────────────────────────────────────────────────────────
TARGET_COLUMN = "Result"
ARTIFACT_DIR = "artifacts"
DATA_FILE_PATH = os.path.join("data", "phisingData.csv")
SCHEMA_FILE_PATH = "schema.yaml"
TRAIN_TEST_SPLIT_RATIO = 0.2
EXPECTED_ACCURACY = 0.6
OVERFITTING_THRESHOLD = 0.05
TRAINING_BUCKET_NAME = "netwworksecurity"

IMPUTER_PARAMS = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

# MongoDB — optional. Set MONGO_DB_URL env var to use instead of local CSV.
MONGO_DB_DATABASE = os.getenv("MONGO_DB_DATABASE", "CHARUKAGUN")
MONGO_DB_COLLECTION = os.getenv("MONGO_DB_COLLECTION", "NetworkData")


# ─── Pipeline Config ───────────────────────────────────────────────────────────
class TrainingPipelineConfig:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_dir = os.path.join(ARTIFACT_DIR, self.timestamp)


class DataIngestionConfig:
    def __init__(self, pipeline_config: TrainingPipelineConfig):
        base = os.path.join(pipeline_config.artifact_dir, "data_ingestion")
        self.train_file_path = os.path.join(base, "train.csv")
        self.test_file_path = os.path.join(base, "test.csv")
        self.train_test_split_ratio = TRAIN_TEST_SPLIT_RATIO


class DataValidationConfig:
    def __init__(self, pipeline_config: TrainingPipelineConfig):
        base = os.path.join(pipeline_config.artifact_dir, "data_validation")
        self.valid_train_file_path = os.path.join(base, "train.csv")
        self.valid_test_file_path = os.path.join(base, "test.csv")
        self.drift_report_file_path = os.path.join(base, "drift_report.yaml")


class DataTransformationConfig:
    def __init__(self, pipeline_config: TrainingPipelineConfig):
        base = os.path.join(pipeline_config.artifact_dir, "data_transformation")
        self.transformed_train_file_path = os.path.join(base, "train.npy")
        self.transformed_test_file_path = os.path.join(base, "test.npy")
        self.preprocessor_file_path = os.path.join(base, "preprocessor.pkl")


class ModelTrainerConfig:
    def __init__(self, pipeline_config: TrainingPipelineConfig):
        base = os.path.join(pipeline_config.artifact_dir, "model_trainer")
        self.trained_model_file_path = os.path.join(base, "model.pkl")
        self.expected_accuracy = EXPECTED_ACCURACY
        self.overfitting_threshold = OVERFITTING_THRESHOLD


# ─── Artifact Dataclasses ──────────────────────────────────────────────────────
@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    preprocessor_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetric:
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric: ClassificationMetric
    test_metric: ClassificationMetric
