from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from real_estate_price_predictor.constants import *
from real_estate_price_predictor.utils.common import read_yaml, create_directories,save_object
from real_estate_price_predictor.entity.config_entity import ModelTrainingConfig
import os
import pandas as pd
import numpy as np
from dvclive import Live

class ModelTraining:
    def __init__(self,config=ModelTrainingConfig):
        self.config = config
        self.best_params ={}
        self.params ={}
        self.metric = {}

        
    
    def model_training(self):
        X_test_data = pd.read_csv(self.config.X_test_scaled_data_file)
        X_train_data  = pd.read_csv(self.config.X_train_scaled_data_file)
        y_train = pd.read_csv(self.config.Y_train_data_file)
        y_test = pd.read_csv(self.config.Y_test_data_file)
        list_of_models = self.config.params_list_of_models
        r2_score_of_models=[]
        adjusted_r2_score =[]
        mse=[]
        for i  in list_of_models:
            if i == 'Linear Regression':
                model = LinearRegression()
            elif i == 'Ridge Regression':
                model = Ridge()
            elif i == 'Polynomial Regression':
                model = Pipeline([(self.config.params_polynomial_type, PolynomialFeatures(degree=self.config.params_polynomial_degree)),(self.config.params_polynomial_model, LinearRegression())])
            elif i == 'SVR':
                model = SVR(kernel=self.config.params_kernel, C=self.config.params_C)
            elif i == 'Random Forrest Regressor':
                model = RandomForestRegressor(n_estimators=self.config.params_n_estimators)
            elif i == 'AdaBoost Regressor':
                model = AdaBoostRegressor()
            elif i == 'Gradient Boosting Regressor':
                model = GradientBoostingRegressor()
            elif i == 'XGBRegressor':
                model = XGBRegressor()
            else:
                model = DecisionTreeRegressor(max_depth=self.config.params_max_depth)
        # Train the model on the training data

            model.fit(X_train_data, y_train)

        # Make predictions on the testing data

            y_pred = model.predict(X_test_data)

        # Evaluate the model performance (e.g., R-squared, Mean Squared Error)
            
            r2 = r2_score(y_test, y_pred)
            r2_score_of_models.append(r2)

        # Calculate the adjusted R²

            n = X_test_data.shape[0]  # Number of observations (samples) in the testing set
            p = X_test_data.shape[1]  # Number of features in the model
            adj_r2_score = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            adjusted_r2_score.append(adj_r2_score)
            mean_square_error = mean_squared_error(y_test, y_pred)
            mse.append(mean_square_error)
            self.metric[i+'_Adjusted_R2_Score'] = adj_r2_score
            self.metric[i+'_R2_Score'] = r2
            self.metric[i+'_Mean_Squared_Error'] = mean_square_error
        data = {'Models': list_of_models, 'Adjusted_R2_Score': adjusted_r2_score, 'R2_Score': r2_score_of_models , 'Mean_Squared_Error': mse}
        performance_metrics = pd.DataFrame.from_dict(data)
        performance_metrics.set_index('Models', inplace = False)
        return performance_metrics
    
    def hyperparameter_tuning(self,performance_metrics:pd.DataFrame):
        X_test_data = pd.read_csv(self.config.X_test_scaled_data_file)
        X_train_data  = pd.read_csv(self.config.X_train_scaled_data_file)
        y_train = pd.read_csv(self.config.Y_train_data_file)
        y_test = pd.read_csv(self.config.Y_test_data_file)
        list_of_models_for_hyper_parameter_tuning = self.config.params_list_of_models_for_hyper_parameter_tuning
        r2_score_of_models_hyper=[]
        adjusted_r2_score_hyper =[]
        mse_hyper = []
        for i in list_of_models_for_hyper_parameter_tuning:
            if i == 'Hyper Parameter Ridge Regression':
                random_grid = {'alpha': [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)],
                            'solver': self.config.params_ridge_regression_solver,
                            'tol': self.config.params_tol}
                model = Ridge()
                rf_randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=self.config.params_number_of_iteration,cv=self.config.params_cv,verbose=self.config.params_verbose,
                                            random_state=self.config.params_random_state_for_randomised_cv,n_jobs=self.config.params_n_jobs)
            elif i == 'Hyper Parameter Support Vector Regression':
                random_grid = {'kernel': [self.config.params_kernel],
                            'C': [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)] + [int(x) for x in np.arange(1, 11)],
                            'epsilon': [float(x) for x in np.linspace(start = 0.01, stop = 0.1, num = 10)] + [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)],
                            'gamma': self.config.params_SVR_gamma
                            }
                model = SVR()
                rf_randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=self.config.params_number_of_iteration,cv=self.config.params_cv,verbose=self.config.params_verbose,
                                            random_state=self.config.params_random_state_for_randomised_cv,n_jobs=self.config.params_n_jobs)
            elif i == 'Hyper Parameter Randomn Forrest Regression':
                random_grid = {'max_features':self.config.params_max_features,
                    'n_estimators': self.config.params_hyper_n_estimators
                    }
                model = RandomForestRegressor()
                rf_randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=self.config.params_number_of_iteration,cv=self.config.params_cv,verbose=self.config.params_verbose,
                                            random_state=self.config.params_random_state_for_randomised_cv,n_jobs=self.config.params_n_jobs)
            elif i == 'Hyper Parameter Gradient Boost Regression':
                random_grid = {
                            'learning_rate':self.config.params_gradient_boost_learning_rate,
                            'subsample':self.config.params_subsample,
                            'n_estimators': self.config.params_hyper_n_estimators
                    }
                model = GradientBoostingRegressor()
                rf_randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=self.config.params_number_of_iteration,cv=self.config.params_cv,verbose=self.config.params_verbose,
                                            random_state=self.config.params_random_state_for_randomised_cv,n_jobs=self.config.params_n_jobs)
            elif i == 'Hyper Parameter XGBoost Regression':
                random_grid = {
                            'learning_rate':self.config.params_xgboost_learning_rate,
                            'n_estimators': self.config.params_hyper_n_estimators
                    }
                model = XGBRegressor()
                rf_randomcv=RandomizedSearchCV(estimator=model,param_distributions=random_grid,n_iter=self.config.params_number_of_iteration,cv=self.config.params_cv,verbose=self.config.params_verbose,
                                            random_state=self.config.params_random_state_for_randomised_cv,n_jobs=self.config.params_n_jobs)
            rf_randomcv.fit(X_train_data,y_train)
            best_random_grid=rf_randomcv.best_estimator_
            y_pred=best_random_grid.predict(X_test_data)
            self.best_params[i] = rf_randomcv.best_estimator_
            # Evaluate the model performance (e.g., R-squared, Mean Squared Error)

            r2_hyper = r2_score(y_test, y_pred)
            r2_score_of_models_hyper.append(r2_hyper)

            # Calculate the adjusted R²

            n = X_test_data.shape[0]  # Number of observations (samples) in the testing set
            p = X_test_data.shape[1]  # Number of features in the model
            adjusted_r2_hyper = 1 - (1 - r2_hyper) * (n - 1) / (n - p - 1)
            adjusted_r2_score_hyper.append(adjusted_r2_hyper)
            mse = mean_squared_error(y_test, y_pred)
            mse_hyper.append(mse)
            self.params[i+"_Params"]=best_random_grid.get_params()
            self.metric[i+'_Adjusted_R2_Score'] = adjusted_r2_hyper
            self.metric[i+'_R2_Score'] = r2_hyper
            self.metric[i+'_Mean_Squared_Error'] = mse
        data = {'Models': list_of_models_for_hyper_parameter_tuning, 'Adjusted_R2_Score': adjusted_r2_score_hyper, 'R2_Score': r2_score_of_models_hyper , 'Mean_Squared_Error': mse_hyper}
        performance_metrics_hyper = pd.DataFrame.from_dict(data)
        performance_metrics_hyper.set_index('Models', inplace = False)
        combined_results = pd.concat([performance_metrics,performance_metrics_hyper],axis=0)
        combined_results = combined_results.sort_values(['Adjusted_R2_Score'],ascending=False)
        return combined_results
    
    def save_the_best_model(self,combined_performance_metrics:pd.DataFrame):
        best_model_name = combined_performance_metrics.sort_values(['Adjusted_R2_Score'],ascending=False).head(1)['Models'].values[0]
        if best_model_name == 'Hyper Parameter Ridge Regression':
            model = self.best_params[best_model_name]
        elif best_model_name == 'Hyper Parameter Support Vector Regression':
            model = self.best_params[best_model_name]
        elif best_model_name == 'Hyper Parameter Randomn Forrest Regression':
            model = self.best_params[best_model_name]
        elif best_model_name == 'Hyper Parameter Gradient Boost Regression':
            model = self.best_params[best_model_name]
        elif best_model_name == 'Hyper Parameter XGBoost Regression':
            model = self.best_params[best_model_name]
        elif best_model_name == 'linear Regression':
            model = LinearRegression()
        elif best_model_name == 'Ridge Regression':
            model = Ridge()
        elif best_model_name == 'Polynomial Regression':
            model = Pipeline([(self.config.params_polynomial_type, PolynomialFeatures(degree=self.config.params_polynomial_degree)),(self.config.params_polynomial_model, LinearRegression())])
        elif best_model_name == 'SVR':
            model = SVR(kernel=self.config.params_kernel, C=self.config.params_C)
        elif best_model_name == 'Random Forrest Regressor':
            model = RandomForestRegressor(n_estimators=self.config.params_n_estimators)
        elif best_model_name == 'AdaBoost Regressor':
            model = AdaBoostRegressor()
        elif best_model_name == 'Gradient Boosting Regressor':
            model = GradientBoostingRegressor()
        elif best_model_name == 'XGBRegressor':
            model = XGBRegressor()
        else:
            model = DecisionTreeRegressor(max_depth=self.config.params_max_depth)

        combined_performance_metrics.to_csv(self.config.performance_metrics_file_path)
        best_model_name = best_model_name.replace(" ","") + '.pkl'
        file_path = os.path.join(self.config.root_dir,best_model_name)
        save_object(file_path,model)
        with Live() as live:
            for key, value in self.metric.items():
                live.log_metric(key,value)
            live.log_params(self.params)
            live.log_artifact(file_path,type="model")
            live.next_step()