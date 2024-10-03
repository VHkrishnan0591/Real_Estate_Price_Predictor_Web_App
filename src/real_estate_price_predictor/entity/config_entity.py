from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    bucket_name: str
    local_data_file: Path
    s3_file_path: str
    amazon_s3_access_keys:str
    params_amazon_service_name:str

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir:Path
    data_file:Path
    transformed_data:Path
    params_discrete_feature: list
    params_Id_column: list
    params_categorical_stratergy:str
    params_numerical_stratergy:str
    params_fill_value:str
    params_target_label:str
    params_rare_categorical_variable:str
    params_ordinal_categorical_feature:list
    params_nominal_categorical_feature:list

@dataclass(frozen=True)
class FeatureSelectionConfig:
    root_dir:Path
    transformed_data_file:Path
    X_train_data_file:Path
    X_test_data_file:Path
    Y_train_data_file:Path
    Y_test_data_file:Path
    params_target_label:str
    params_Id_column: list
    params_alpha_for_lasso: int
    params_random_state_for_lasso:int
    params_percentile_for_mutual_info:int
    params_test_size:int
    params_random_state_for_train_test_split:int

@dataclass(frozen=True)
class FeatureScalingConfig:
    root_dir:Path
    X_train_data_file:Path
    X_test_data_file:Path
    X_train_scaled_data_file:Path
    X_test_scaled_data_file:Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    X_train_scaled_data_file: Path
    X_test_scaled_data_file: Path
    Y_train_data_file: Path
    Y_test_data_file: Path
    performance_metrics_file_path: Path
    params_list_of_models:list
    params_polynomial_type:str
    params_polynomial_model:str
    params_polynomial_degree:int
    params_kernel:str
    params_C:int
    params_n_estimators:int
    params_max_depth:int
    params_number_of_iteration: int
    params_cv: int
    params_verbose: int
    params_random_state_for_randomised_cv: int
    params_n_jobs: int
    params_list_of_models_for_hyper_parameter_tuning: list
    params_ridge_regression_solver: list
    params_tol: list
    params_SVR_gamma: list
    params_max_features: list
    params_hyper_n_estimators: list
    params_gradient_boost_learning_rate: list
    params_subsample: list
    params_xgboost_learning_rate: list