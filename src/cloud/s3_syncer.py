import os
import boto3
from botocore.exceptions import ClientError

from src.logger import logging


class S3Syncer:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")

    def sync_to_s3(self, local_path: str, s3_prefix: str) -> None:
        """Upload a local file or directory to S3."""
        if os.path.isdir(local_path):
            for root, _, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative = os.path.relpath(file_path, local_path)
                    s3_key = "/".join([s3_prefix.rstrip("/"), relative.replace("\\", "/")])
                    logging.info(f"Uploading {file_path} → s3://{self.bucket_name}/{s3_key}")
                    self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
        else:
            s3_key = "/".join([s3_prefix.rstrip("/"), os.path.basename(local_path)])
            logging.info(f"Uploading {local_path} → s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)

    def sync_from_s3(self, s3_prefix: str, local_path: str) -> None:
        """Download all objects under an S3 prefix to a local directory."""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative = os.path.relpath(key, s3_prefix)
                dest = os.path.join(local_path, relative)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                logging.info(f"Downloading s3://{self.bucket_name}/{key} → {dest}")
                self.s3_client.download_file(self.bucket_name, key, dest)
