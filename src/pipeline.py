import sys

from src.exception import NetworkSecurityException
from src.logger import logging
from src.config import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def run_pipeline(self):
        try:
            pipeline_config = TrainingPipelineConfig()

            logging.info("=" * 50)
            logging.info("=== STEP 1: Data Ingestion ===")
            ingestion = DataIngestion(DataIngestionConfig(pipeline_config))
            ingestion_artifact = ingestion.initiate_data_ingestion()

            logging.info("=== STEP 2: Data Validation ===")
            validation = DataValidation(ingestion_artifact, DataValidationConfig(pipeline_config))
            validation_artifact = validation.initiate_data_validation()

            logging.info("=== STEP 3: Data Transformation ===")
            transformation = DataTransformation(
                validation_artifact, DataTransformationConfig(pipeline_config)
            )
            transformation_artifact = transformation.initiate_data_transformation()

            logging.info("=== STEP 4: Model Training ===")
            trainer = ModelTrainer(transformation_artifact, ModelTrainerConfig(pipeline_config))
            trainer_artifact = trainer.initiate_model_trainer()

            logging.info("=" * 50)
            logging.info(f"Pipeline complete — model: {trainer_artifact.trained_model_file_path}")
            return trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
