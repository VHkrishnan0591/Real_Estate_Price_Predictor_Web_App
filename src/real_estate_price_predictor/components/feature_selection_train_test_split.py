from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from real_estate_price_predictor import logger
import pandas as pd
import os
from real_estate_price_predictor.entity.config_entity import FeatureSelectionConfig

class FeatureSelectionAndTrainTestSplit:
    
    def __init__(self,config=FeatureSelectionConfig):
        self.config = config

    def lasso_feature_selection(self):
        if os.path.exists(self.config.transformed_data_file):
            dataset = pd.read_csv(self.config.transformed_data_file)
            X = dataset.drop([self.config.params_Id_column[0],self.config.params_target_label,],axis=1)
            Y = dataset[[self.config.params_target_label]]
            feature_sel_model = SelectFromModel(Lasso(alpha=self.config.params_alpha_for_lasso, random_state=self.config.params_random_state_for_lasso)) # remember to set the seed, the random state in this function
            feature_sel_model.fit(X, Y)
            selected_feat = X.columns[(feature_sel_model.get_support())]
            return X[selected_feat]
        else: logger.info(f">>>>>> transformation data file is not present <<<<<<")
        
    
    def mutual_information_feature_selection(self):
        if os.path.exists(self.config.transformed_data_file):
            dataset = pd.read_csv(self.config.transformed_data_file)
            X = dataset.drop([self.config.params_Id_column[0],self.config.params_target_label],axis=1)
            Y = dataset[[self.config.params_target_label]]
            selected_top_columns = SelectPercentile(mutual_info_regression, percentile=20)
            selected_top_columns.fit(X, Y)
            selected_feature = X.columns[selected_top_columns.get_support()]
            return X[selected_feature]
        else: logger.info(f">>>>>> transformation data file is not present <<<<<<")

    def test_train_split(self,dataset:pd.DataFrame):
        X = dataset
        if os.path.exists(self.config.transformed_data_file):
            dataset = pd.read_csv(self.config.transformed_data_file)
            Y = dataset[[self.config.params_target_label]]
            X_train,X_test,y_train,y_test =train_test_split(X,Y,test_size=self.config.params_test_size,random_state=self.config.params_random_state_for_train_test_split)
            X_train = pd.DataFrame(X_train,columns=X.columns)
            X_test = pd.DataFrame(X_test,columns=X.columns)
            X_train.to_csv(self.config.X_train_data_file,index=False)
            X_test.to_csv(self.config.X_test_data_file,index=False)
            y_train = pd.DataFrame(y_train,columns=Y.columns)
            y_test = pd.DataFrame(y_test,columns=Y.columns)
            y_train.to_csv(self.config.Y_train_data_file,index=False)
            y_test.to_csv(self.config.Y_test_data_file,index=False)
        else: logger.info(f">>>>>> transformation data file is not present <<<<<<")