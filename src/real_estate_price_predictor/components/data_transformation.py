import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from real_estate_price_predictor.entity.config_entity import DataTransformationConfig

class SeparatingDifferentFeatures:
    def __init__(self,config:DataTransformationConfig):
        self.config = config
    
    def read_data(self):
        dataset = pd.read_csv(self.config.data_file)
        return dataset

# Features with null values

    def features_with_null_values(self, df:pd.DataFrame):
        dataset = df
        features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]
        return features_with_na
    
# Finding both numerical and categorical features with null values

    def num_and_categorical_features_with_na(self,df:pd.DataFrame, categorical:bool):
        numerical_features_with_na =[]
        categorical_features_with_na =[]
        features_with_na = self.features_with_null_values(df)
        for feature in features_with_na:
            if pd.api.types.is_numeric_dtype(df[feature]):
                numerical_features_with_na.append(feature)
            else:
                categorical_features_with_na.append(feature)
        if categorical:
            return categorical_features_with_na
        else: 
            return numerical_features_with_na
    
# Finding all the numerical features

    def total_numerical_features(self,dataset:pd.DataFrame):
        numerical_features = []
        for feature in dataset.columns:
            if dataset[feature].dtypes != 'O':
                numerical_features.append(feature)
        return numerical_features

# Finding year or datatime variable

    def finding_year_feature(self,dataset:pd.DataFrame):
        year_feature = []
        numerical_features = self.total_numerical_features(dataset)
        for feature in numerical_features:
            if 'Yr' in feature or 'Year' in feature:
                year_feature.append(feature)
        return year_feature
        
# Finding Continuous Variable

    def continous_variables(self,df:pd.DataFrame):
        continuous_feature=[]
        numerical_features = self.total_numerical_features(df) 
        year_feature = self.finding_year_feature(df)
        for feature in numerical_features:
            if feature not in self.config.params_discrete_feature+year_feature+self.config.params_Id_column:
                continuous_feature.append(feature)
        return continuous_feature

# Replacing the zeros with 1 to perform log transform

    def replacing_zeros_of_continuous_features(self,dataset:pd.DataFrame):
        continuous_feature = self.continous_variables(dataset)
        for feature in continuous_feature:
            dataset.loc[dataset[feature] == 0, feature] = 1
        return dataset

# Finding the categorical features

    def total_categorical_features(self,dataset:pd.DataFrame):
        return [feature for feature in dataset.columns if dataset[feature].dtypes=='O']

# Handling Missing Values by creating a new category for categroical and with median for numerical

    def filling_missing_values(self, dataset:pd.DataFrame):
        categorical_imputer = SimpleImputer(strategy= self.config.params_categorical_stratergy,fill_value=self.config.params_fill_value)
        numerical_imputer = SimpleImputer(strategy=self.config.params_numerical_stratergy)
        categorical_features = self.total_categorical_features(dataset)
        numerical_features = self.total_numerical_features(dataset)
        dataset[categorical_features] = categorical_imputer.fit_transform(dataset[categorical_features])
        dataset[numerical_features] = numerical_imputer.fit_transform(dataset[numerical_features])
        return dataset
    
# Saving the transformed data

    def save_the_transformed_data(self,dataset:pd.DataFrame):
        dataset.to_csv(self.config.transformed_data,index=False)


# Handling Date time variables

class handling_date_time_variables(BaseEstimator, TransformerMixin,SeparatingDifferentFeatures):
    def __init__(self,config:DataTransformationConfig): # no *args or **kargs 
         super().__init__(config)
    def fit(self, X, y=None):
         return self # nothing else to do
    def transform(self, X, y=None):
         year_features = self.finding_year_feature(X)
         for feature in year_features:
             if feature != 'YrSold':
                 X[feature]=X['YrSold']-X[feature]
         return X

# Transforming the continous features using log transform

class log_transform_of_numeric_variables(BaseEstimator, TransformerMixin,SeparatingDifferentFeatures):
    def __init__(self, config:DataTransformationConfig): # no *args or **kargs
         super().__init__(config)
    def fit(self, X, y=None):
         return self # nothing else to do
    def transform(self, X, y=None):
         continous_features = self.continous_variables(X)
         X = self.replacing_zeros_of_continuous_features(X)
         for feature in continous_features:
             X[feature]=np.log(X[feature])
         return X

# Handling rare categorical features

class handling_rare_categorical_values(BaseEstimator, TransformerMixin,SeparatingDifferentFeatures):
    def __init__(self,config:DataTransformationConfig): # no *args or **kargs
         super().__init__(config)
    def fit(self, X, y=None):
         return self # nothing else to do
    def transform(self, X, y=None):
         categorical_features = self.total_categorical_features(X)
         for feature in categorical_features:
             temp=X.groupby(feature)[self.config.params_target_label].count()/len(X)
             temp_df=temp[temp>0.01].index
             X[feature]=np.where(X[feature].isin(temp_df),X[feature],self.config.params_rare_categorical_variable)
         return X

# Ordinal Catgeorical Features Encoding

class handling_ordinal_categorical_values(BaseEstimator, TransformerMixin,SeparatingDifferentFeatures):
    def __init__(self,config:DataTransformationConfig): # no *args or **kargs
         self.label_ordered_feature ={}
         super().__init__(config)
    def fit(self, X, y=None):
         for feature in self.config.params_ordinal_categorical_feature:
             labels_ordered=X.groupby([feature])[self.config.params_target_label].mean().sort_values().index
             labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
             self.label_ordered_feature[feature] = labels_ordered
         return self
    def transform(self, X, y=None):
         for feature in self.config.params_ordinal_categorical_feature:
           X[feature]=X[feature].map(self.label_ordered_feature[feature])
         return X

# Nominal Categorical Features Encoding

class handling_nominal_categorical_values(BaseEstimator, TransformerMixin,SeparatingDifferentFeatures):
    def __init__(self,config:DataTransformationConfig): # no *args or **kargs
         self.label_nominal_feature ={}
         super().__init__(config)
    def fit(self, X, y=None):
         for feature in self.config.params_nominal_categorical_feature:
             nominal_label=X.groupby([feature])[self.config.params_target_label].mean().to_dict()
             self.label_nominal_feature[feature] = nominal_label
         return self
    def transform(self, X, y=None):
         for feature in self.config.params_nominal_categorical_feature:
           X[feature]=X[feature].map(self.label_nominal_feature[feature])
         return X

# Handling Outliers

class handling_outliers_for_continous_variable(BaseEstimator, TransformerMixin,SeparatingDifferentFeatures):
    def __init__(self,config:DataTransformationConfig):
        self.iqr_boundaries_conitnous_feature ={}
        super().__init__(config)
        
    def fit(self, X, y=None):
        continuous_feature = self.continous_variables(X)
        for feature in continuous_feature:
            IQR=X[feature].quantile(0.75)-X[feature].quantile(0.25)
            lower_bridge=X[feature].quantile(0.25)-(IQR*3)
            upper_bridge=X[feature].quantile(0.75)+(IQR*3)
            self.iqr_boundaries_conitnous_feature[feature] = [lower_bridge,upper_bridge]
        return self
    def transform(self, X, y=None):
        continuous_feature = self.continous_variables(X)
        for feature in continuous_feature:
            lower_bridge, upper_bridge = self.iqr_boundaries_conitnous_feature[feature]
            X.loc[X[feature]<=lower_bridge,feature]=lower_bridge
            X.loc[X[feature]>=upper_bridge,feature]=upper_bridge
        return X