import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import NetworkSecurityException
from src.logger import logging
from src.config import (
    DataIngestionConfig, DataIngestionArtifact,
    DATA_FILE_PATH, MONGO_DB_DATABASE, MONGO_DB_COLLECTION,
)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _load_data(self) -> pd.DataFrame:
        mongo_url = os.getenv("MONGO_DB_URL")
        if mongo_url:
            logging.info("Loading data from MongoDB")
            import pymongo
            import certifi
            client = pymongo.MongoClient(mongo_url, tlsCAFile=certifi.where())
            collection = client[MONGO_DB_DATABASE][MONGO_DB_COLLECTION]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            df.replace({"na": np.nan}, inplace=True)
        else:
            logging.info(f"Loading data from local CSV: {DATA_FILE_PATH}")
            df = pd.read_csv(DATA_FILE_PATH)
        return df

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            df = self._load_data()
            logging.info(f"Dataset loaded: {df.shape}")

            train_set, test_set = train_test_split(
                df,
                test_size=self.config.train_test_split_ratio,
                random_state=42,
            )

            os.makedirs(os.path.dirname(self.config.train_file_path), exist_ok=True)
            train_set.to_csv(self.config.train_file_path, index=False)
            test_set.to_csv(self.config.test_file_path, index=False)

            logging.info(f"Train: {train_set.shape}, Test: {test_set.shape}")
            return DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
