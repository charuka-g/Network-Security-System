import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

import certifi
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, BulkWriteError

import pandas as pd
import numpy as np

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val or not val.strip():
        raise NetworkSecurityException(f"Missing required environment variable: {name}", sys)
    return val

def get_mongo_client(uri: str) -> MongoClient:
    """
    Create a secure MongoClient using certifi CA bundle.
    Works with mongodb+srv:// URIs and direct mongodb:// URIs.
    """
    return MongoClient(
        uri,
        tls=True,                          # be explicit
        tlsCAFile=certifi.where(),         # <- key fix for macOS/Python cert chain
        serverSelectionTimeoutMS=15000     # faster fail when handshake/DNS is broken
    )

class NetworkDataExtract:
    def __init__(self):
        try:
            # validate env early so failures are clear
            self.mongo_uri = _require_env("MONGO_DB_URL")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_convertor(self, file_path: str):
        try:
            df = pd.read_csv(file_path)

            # Replace NaN with None so Mongo can store them as nulls
            df = df.replace({np.nan: None})

            # Faster and simpler than the transpose+to_json route
            records = df.to_dict(orient="records")
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database: str, collection: str) -> int:
        try:
            client = get_mongo_client(self.mongo_uri)

            # Quick connectivity check (helps surface TLS/DNS/IP-allowlist issues early)
            client.admin.command("ping")

            db = client[database]
            col = db[collection]

            # ordered=False lets Mongo continue past duplicate-key rows if any
            result = col.insert_many(records, ordered=False)
            return len(result.inserted_ids)

        except BulkWriteError as bwe:
            # If you expect dupes, you can still return the count of successful inserts
            logging.error(f"Bulk write error: {bwe.details}")
            inserted = len(bwe.details.get("writeErrors", []))
            raise NetworkSecurityException(bwe, sys) from bwe

        except ServerSelectionTimeoutError as e:
            # Most common handshake/DNS/allowlist error path
            raise NetworkSecurityException(
                f"Cannot connect to MongoDB: {e}. "
                f"Check: (1) IP allowlist in Atlas, (2) URI uses mongodb+srv, "
                f"(3) network/proxy SSL inspection, (4) system certs.", sys
            ) from e

        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__ == '__main__':
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "CHARUKAGUN"
    COLLECTION = "NetworkData"

    try:
        extractor = NetworkDataExtract()
        records = extractor.csv_to_json_convertor(FILE_PATH)
        print(f"Loaded {len(records)} records from CSV")
        n = extractor.insert_data_mongodb(records, DATABASE, COLLECTION)
        print(f"Inserted {n} records")
    except Exception as e:
        # If your custom exception already logs, this print can be reduced.
        print(e)
        raise
