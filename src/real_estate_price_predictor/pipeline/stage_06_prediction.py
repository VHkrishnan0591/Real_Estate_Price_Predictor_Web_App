from real_estate_price_predictor.config.configuration import ConfigurationManager
from real_estate_price_predictor import logger
from real_estate_price_predictor.components.predict import Predict
import pandas as pd

STAGE_NAME = 'Predict Stage'

class PredictPipeline:
    def __init__(self):
        pass

    def main(self,dataframe:pd.DataFrame):
        config = ConfigurationManager()
        predict_config = config.get_predict_config()
        predict = Predict(config=predict_config)
        result = predict.predict(dataframe)
        return result


