from real_estate_price_predictor.constants import *
from real_estate_price_predictor.utils.common import read_yaml, create_directories
from real_estate_price_predictor.entity.config_entity import DataIngestionConfig, DataTransformationConfig, FeatureScalingConfig, FeatureSelectionConfig, ModelTrainingConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            bucket_name=config.bucket_name,
            local_data_file=config.local_data_file,
            s3_file_path=config.s3_file_path, 
            amazon_s3_access_keys = config.amazon_s3_access_keys,
            params_amazon_service_name= self.params.amazon_service_name
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            data_file = config.data_file,
            transformed_data = config.transformed_data_file,
            params_discrete_feature = self.params.discrete_feature,
            params_Id_column = self.params.Id_column,
            params_categorical_stratergy = self.params.categorical_stratergy,
            params_numerical_stratergy = self.params.numerical_stratergy,
            params_fill_value = self.params.fill_value,
            params_target_label = self.params.target_label,
            params_rare_categorical_variable = self.params.rare_categorical_variable,
            params_ordinal_categorical_feature = self.params.ordinal_categorical_feature,
            params_nominal_categorical_feature = self.params.nominal_categorical_feature
        )

        return data_transformation_config
    
    def get_feature_selection(self) -> FeatureSelectionConfig:
        config = self.config.feature_selection

        create_directories([config.root_dir])

        feature_selection_config = FeatureSelectionConfig(
            root_dir=config.root_dir,
            transformed_data_file = config.transformed_data_file,
            X_train_data_file = config.X_train_data_file,
            X_test_data_file =  config.X_test_data_file,
            Y_train_data_file= config.Y_train_data_file,
            Y_test_data_file = config.Y_test_data_file,
            params_target_label = self.params.target_label,
            params_Id_column = self.params.Id_column,
            params_alpha_for_lasso = self.params.alpha_for_lasso,
            params_random_state_for_lasso = self.params.random_state_for_lasso,
            params_percentile_for_mutual_info = self.params.percentile_for_mutual_info,
            params_test_size = self.params.test_size,
            params_random_state_for_train_test_split = self.params.random_state_for_train_test_split
            
        )

        return feature_selection_config
    
    def get_feature_scaling(self) -> FeatureScalingConfig:
        config = self.config.feature_scaling

        create_directories([config.root_dir])

        feature_scaling_config = FeatureScalingConfig(
            root_dir=config.root_dir,
            X_train_data_file = config.X_train_data_file,
            X_test_data_file = config.X_test_data_file,
            X_train_scaled_data_file = config.X_train_scaled_data_file,
            X_test_scaled_data_file =  config.X_test_scaled_data_file   
        )

        return feature_scaling_config
    
    def get_model_training(self) -> ModelTrainingConfig:
        config = self.config.model_training

        create_directories([config.root_dir])

        model_training_config= ModelTrainingConfig(
            root_dir = config.root_dir,
            X_train_scaled_data_file = config.X_train_scaled_data_file,
            X_test_scaled_data_file = config.X_test_scaled_data_file,
            Y_train_data_file = config.Y_train_data_file,
            Y_test_data_file = config.Y_test_data_file,
            performance_metrics_file_path = config.performance_metrics_file_path,
            params_list_of_models = self.params.list_of_models,
            params_polynomial_type = self.params.polynomial_type,
            params_polynomial_model = self.params.polynomial_model,
            params_polynomial_degree = self.params.polynomial_degree,
            params_kernel = self.params.kernel,
            params_C = self.params.C,
            params_n_estimators = self.params.n_estimators,
            params_max_depth = self.params.max_depth,
            params_number_of_iteration = self.params.number_of_iteration,
            params_cv = self.params.cv,
            params_verbose = self.params.verbose,
            params_random_state_for_randomised_cv = self.params.random_state_for_randomised_cv,
            params_n_jobs = self.params.n_jobs,
            params_list_of_models_for_hyper_parameter_tuning = self.params.list_of_models_for_hyper_parameter_tuning,
            params_ridge_regression_solver = self.params.ridge_regression_solver,
            params_tol = self.params.tol,
            params_SVR_gamma = self.params.SVR_gamma,
            params_max_features = self.params.max_features,
            params_hyper_n_estimators = self.params.hyper_n_estimators,
            params_gradient_boost_learning_rate = self.params.gradient_boost_learning_rate,
            params_subsample = self.params.subsample,
            params_xgboost_learning_rate = self.params.xgboost_learning_rate

        )

        return model_training_config