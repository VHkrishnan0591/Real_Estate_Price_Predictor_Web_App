from real_estate_price_predictor.config.configuration import ConfigurationManager
from real_estate_price_predictor import logger
from real_estate_price_predictor.components.feature_scaling import FeatureScaling
import pandas as pd

STAGE_NAME = 'Feature Scaling Stage'

class FeatureScalingPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feature_scaling_config = config.get_feature_scaling()
        logger.info(f">>>>>> stage Feature Scaling started <<<<<<")
        feature_scaling = FeatureScaling(config=feature_scaling_config)
        logger.info(f">>>>>> stage Feature Scaling ended <<<<<<")
        feature_scaling.min_max_scaler(pd.DataFrame())


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureScalingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e