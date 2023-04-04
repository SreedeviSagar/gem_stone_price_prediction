import sys
import os

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
#from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from imblearn.over_sampling import RandomOverSampler

from dataclasses import dataclass

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns=['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_columns=['cut', 'color', 'clarity']

            logging.info(f'numerical columns:{numerical_columns}')
            logging.info(f'categorical columns:{categorical_columns}')

            cut_enc = ["Fair","Good","Very Good","Premium","Ideal"]
            clarity_enc = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
            color_enc = ["J","I","H","G","F","E","D"]
           
            num_pipeline=Pipeline(steps=[('scaler',StandardScaler())])
            cat_pipeline=Pipeline(steps=[('ord_enc',OrdinalEncoder(categories=[cut_enc,color_enc,clarity_enc])),
                             ('scaler',StandardScaler())])
            preprocessor=ColumnTransformer([('num_pipeline',num_pipeline,numerical_columns),('cat_pipeline',cat_pipeline,categorical_columns)])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train data and test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformation_object()

            target_column_name='price'
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info('Forming train array and test array')
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)
            return(train_arr,test_arr)
            

        except Exception as e:
            raise CustomException(e,sys)