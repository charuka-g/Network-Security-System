import os
import sys

from src.exception import NetworkSecurityException
from src.logger import logging
from src.config import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    TRAINING_BUCKET_NAME,
)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.cloud.s3_syncer import S3Syncer


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

            if os.getenv("AWS_ACCESS_KEY_ID"):
                logging.info("=== STEP 5: Syncing artifacts to S3 ===")
                syncer = S3Syncer(TRAINING_BUCKET_NAME)
                syncer.sync_to_s3(
                    pipeline_config.artifact_dir,
                    f"artifacts/{pipeline_config.timestamp}",
                )
                syncer.sync_to_s3("models", "models")
                logging.info("Artifacts synced to S3")

            logging.info("=" * 50)
            logging.info(f"Pipeline complete — model: {trainer_artifact.trained_model_file_path}")
            return trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
