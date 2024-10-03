from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from real_estate_price_predictor.entity.config_entity import FeatureScalingConfig
import os
from pathlib import Path
class FeatureScaling():
    def __init__(self,config = FeatureScalingConfig):
        self.config =  config
    
    def read_csv_file(self,path:Path):
        dataset = pd.read_csv(path)
        return dataset 
    
    def min_max_scaler(self,dataset:pd.DataFrame):
        X_train = self.read_csv_file(self.config.X_train_data_file)
        X_test = self.read_csv_file(self.config.X_test_data_file)
        scaler=MinMaxScaler()
        scaler.fit(X_train)
        if dataset.isnull().all().all():
            if (os.path.exists(self.config.X_train_data_file)) and (os.path.exists(self.config.X_test_data_file) ):
                X_train_data = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
                X_test_data = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
                X_train_data.to_csv(self.config.X_train_scaled_data_file,index=False)
                X_test_data.to_csv(self.config.X_test_scaled_data_file,index=False)
        else: 
            dataframe = pd.DataFrame(scaler.transform(dataset), columns=dataset.columns)
            return dataframe