import os
import urllib.request as request
import boto3
from pathlib import Path
from real_estate_price_predictor import logger
from real_estate_price_predictor.utils.common import get_size
from real_estate_price_predictor.entity.config_entity import DataIngestionConfig
import pandas as pd

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def ingestion(self):
        if not os.path.exists(self.config.local_data_file):
            df = pd.read_csv('data.csv')
            df.to_csv(self.config.local_data_file, index=False)
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}") 
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            df = pd.read_csv(self.config.amazon_s3_access_keys)
            s3 = boto3.client(self.config.params_amazon_service_name,aws_access_key_id=df[df.columns[0]][0], aws_secret_access_key=df[df.columns[1]][0])
            s3.download_file(self.config.bucket_name, self.config.s3_file_path, self.config.local_data_file)
            logger.info(f"{self.config.local_data_file} download! with following info")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}") 