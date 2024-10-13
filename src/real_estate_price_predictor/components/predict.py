from real_estate_price_predictor.entity.config_entity import PredictConfig
import os
import pandas as pd
import numpy as np
from real_estate_price_predictor.utils.common import load_object


class Predict:
    
    def __init__(self, config = PredictConfig):
        self.config = config
    
    def predict(self,dataframe:pd.DataFrame):
        date_time_transformer = load_object(self.config.date_time_handler_model_file)
        log_transformer = load_object(self.config.log_transformer_model_file)
        rare_categorical_model = load_object(self.config.rare_categorical_handler_file)
        ordinal_encoder = load_object(self.config.ordinal_encoder_model_file)
        nominal_encoder = load_object(self.config.nominal_encoder_model_file)
        remove_outlier_model = load_object(self.config.remove_outlier_model_file)
        feature_scaler = load_object(self.config.feature_scaling_model)
        for i in os.listdir(self.config.best_model_directory):
            if i.__contains__('pkl'):
                model_name = i
        model = load_object(os.path.join(self.config.best_model_directory,model_name))
        dataframe = date_time_transformer.transform(dataframe)
        dataframe = log_transformer.transform(dataframe)
        dataframe = rare_categorical_model.transform(dataframe)
        dataframe = ordinal_encoder.transform(dataframe)
        dataframe = nominal_encoder.transform(dataframe)
        dataframe = remove_outlier_model.transform(dataframe)
        dataframe.drop(['YrSold'],axis=1,inplace= True)
        dataframe = pd.DataFrame(feature_scaler.transform(dataframe), columns=dataframe.columns)
        y_pred = model.predict(dataframe)
        return np.exp(y_pred[0])
