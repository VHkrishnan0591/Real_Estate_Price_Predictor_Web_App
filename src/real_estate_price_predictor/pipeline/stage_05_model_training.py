from real_estate_price_predictor import logger
from real_estate_price_predictor.config.configuration import ConfigurationManager
from real_estate_price_predictor.components.model_training import ModelTraining

STAGE_NAME = 'Model Training Stage'


class ModelTrainingPipeline:

    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training()
        model_training = ModelTraining(config=model_training_config)
        logger.info(f">>>>>> stage Model Training started <<<<<<")
        results = model_training.model_training()
        logger.info(f">>>>>> stage Hyper Parameter Tuning started <<<<<<")
        combined_performance_metrics = model_training.hyperparameter_tuning(results)
        logger.info(f">>>>>> stage Hyper Parameter Tuning ended <<<<<<")
        logger.info(f">>>>>> stage Model Training ended <<<<<<")
        logger.info(f">>>>>> stage saving the best model started <<<<<<")
        model_training.save_the_best_model(combined_performance_metrics)
        logger.info(f">>>>>> stage saving the best model ended <<<<<<")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e