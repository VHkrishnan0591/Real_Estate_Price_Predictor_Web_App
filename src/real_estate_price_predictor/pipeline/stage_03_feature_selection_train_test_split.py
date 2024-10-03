from real_estate_price_predictor.config.configuration import ConfigurationManager
from real_estate_price_predictor import logger
from real_estate_price_predictor.components.feature_selection_train_test_split import FeatureSelectionAndTrainTestSplit

STAGE_NAME = 'Feature Selection and Train Test Split Stage'

class FeatureSelectionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feature_selection_config = config.get_feature_selection()
        feature_selection = FeatureSelectionAndTrainTestSplit(config=feature_selection_config)

        logger.info(f">>>>>> stage Feature Selection started <<<<<<")
        dataset = feature_selection.lasso_feature_selection()
        logger.info(f">>>>>> stage Feature Selection ended <<<<<<")

        logger.info(f">>>>>> stage train test split started <<<<<<")
        feature_selection.test_train_split(dataset)
        logger.info(f">>>>>> stage train test split started <<<<<<")
    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureSelectionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e