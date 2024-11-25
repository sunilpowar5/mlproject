import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer ## for missing values
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = [
                'gender','race_ethnicity',
                'parental_level_of_education','lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(

                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                    ]
            )
            logging.info("Numerical Standardization completed")

            cat_pipeline = Pipeline(

                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder(sparse=False)),
                    ('scaler',StandardScaler(with_mean=False))
                    ]
            )
            logging.info('categorical encoding colpleted')

            preprocessor = ColumnTransformer(

                [
                    ('numerical_pipeline',num_pipeline,numerical_columns),
                    ('categorical_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)    

            logging.info('Read train and test completed')

            logging.info('obtraining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_col_name = 'math_score'
            numerical_col_names = ['writing_score','reading_score']

            input_feature_train_df = train_df.drop(target_col_name,axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(target_col_name,axis=1)
            target_feature_test_df = test_df[target_col_name]

            logging.info('applying preprocessing object on training and testing dataframes.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr =  np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)

            ]

            test_arr =  np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
                
            ]

            logging.info('Saved preprocessing objects')

            save_object(

                filepath = self.data_tranformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )   ## utils

            return(
                train_arr,
                test_arr
            )

        except Exception as e:
            raise CustomException(e,sys)