from real_estate_price_predictor.config.configuration import ConfigurationManager
from real_estate_price_predictor import logger
from real_estate_price_predictor.components.data_transformation import *


STAGE_NAME = 'Data Transformation Stage'

class DataTransformationPipeline:

    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = SeparatingDifferentFeatures(config=data_transformation_config)

        logger.info(f">>>>>> stage Read the data started <<<<<<")

        dataset = data_transformation.read_data()

        logger.info(f">>>>>> stage Read the data ended <<<<<<")
         
        logger.info(f">>>>>> stage Handling Null Values started <<<<<<")

        dataset = data_transformation.filling_missing_values(dataset)

        logger.info(f">>>>>> stage Handling Null Values Ended <<<<<<")

        logger.info(f">>>>>> stage Handling date time variables started <<<<<<")

        date_time_variables = handling_date_time_variables(config=data_transformation_config)
        dataset = date_time_variables.transform(dataset)

        logger.info(f">>>>>> stage Handling date time variables ended <<<<<<")

        logger.info(f">>>>>> stage Transforming the continous variables using logrithmic transform started <<<<<<")

        log_transform = log_transform_of_numeric_variables(config=data_transformation_config)
        dataset = log_transform.transform(dataset)

        logger.info(f">>>>>> stage Transforming the continous variables using logrithmic transform ended <<<<<<")

        logger.info(f">>>>>> stage Handling rare categorical variable started <<<<<<")

        rare_categorical_values = handling_rare_categorical_values(config=data_transformation_config)
        rare_categorical_values.fit(dataset)
        dataset = rare_categorical_values.transform(dataset)    

        logger.info(f">>>>>> stage Handling rare categorical variable ended <<<<<<")

        logger.info(f">>>>>> stage Encoding the ordinal categorical features using (Target Guided Encoding)  started <<<<<<")

        ordinal_features = handling_ordinal_categorical_values(config=data_transformation_config)
        ordinal_features.fit(dataset)
        dataset = ordinal_features.transform(dataset)

        logger.info(f">>>>>> stage Encoding the ordinal categorical features using (Target Guided Encoding)  ended <<<<<<")

        logger.info(f">>>>>> stage Encoding the nominal categorical features using (Mean Encoding) started <<<<<<")

        nominal_features = handling_nominal_categorical_values(config=data_transformation_config)
        nominal_features.fit(dataset)
        dataset = nominal_features.transform(dataset) 

        logger.info(f">>>>>> stage Encoding the nominal categorical features using (Mean Encoding) ended <<<<<<") 

        logger.info(f">>>>>> stage Removing the Outliers in Continous feature started <<<<<<")

        Removal_of_outlier = handling_outliers_for_continous_variable(config=data_transformation_config)
        Removal_of_outlier.fit(dataset)
        dataset = Removal_of_outlier.transform(dataset)

        logger.info(f">>>>>> stage Removing the Outliers in Continous feature ended <<<<<<")

        logger.info(f">>>>>> stage Saving the transformed data started <<<<<<")

        data_transformation.save_the_transformed_data(dataset)

        logger.info(f">>>>>> stage Saving the transformed data ended <<<<<<")

        logger.info(f">>>>>> stage Saving the transformation model started <<<<<<")

        dict_of_preporcessing_models ={
                               'rare_categorical_values_handler':rare_categorical_values,
                               'date_time_variables': date_time_variables,
                               'log_transformer': log_transform,
                               'ordinal_encoder':ordinal_features,
                               'nominal_encoder':nominal_features,
                               'remove_outliers_transformer':Removal_of_outlier}
        data_transformation.save_the_model(dict_of_preporcessing_models)

        logger.info(f">>>>>> stage Saving the transformation model ended <<<<<<")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e