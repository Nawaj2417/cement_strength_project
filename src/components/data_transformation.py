import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            numerical_cols = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 
                              'Coarse Aggregate', 'Fine Aggregate', 'Age (day)']

            logging.info('Pipeline Initiated')

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler()),
                ]
            )
            logging.info(f"Numerical columns: {numerical_cols}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_cols),
                ]
            )

            logging.info('Pipeline completed')
            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Original Train DataFrame columns: %s", train_df.columns.tolist())
            logging.info("Original Test DataFrame columns: %s", test_df.columns.tolist())

            # Clean column names
            train_df = self.clean_column_names(train_df)
            test_df = self.clean_column_names(test_df)
            print(test_df)

            logging.info("Cleaned Train DataFrame columns: %s", train_df.columns.tolist())
            logging.info("Cleaned Test DataFrame columns: %s", test_df.columns.tolist())

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "Concrete compressive strength"
            print(test_df.columns)

            # Check for the target column in train_df
            if target_column_name not in train_df.columns:
                logging.error(f"Column '{target_column_name}' not found in train_df. Available columns are: {train_df.columns.tolist()}")
                raise KeyError(f"Column '{target_column_name}' not found in train_df.")
     
            # Check for the target column in test_df
            if target_column_name not in test_df.columns:
                logging.error(f"Column '{target_column_name}' not found in test_df. Available columns are: {test_df.columns.tolist()}")
                raise KeyError(f"Column '{target_column_name}' not found in test_df.")

            logging.info("Target column is present in both DataFrames")

            # Log data types and first few rows to check for any issues
            logging.info("Train DataFrame info: %s", train_df.info())
            logging.info("Test DataFrame info: %s", test_df.info())
            logging.info("Train DataFrame Head: %s", train_df.head())
            logging.info("Test DataFrame Head: %s", test_df.head())

            # Drop the target column and separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error(f"Exception occurred in initiate_data_transformation: {e}")
            raise CustomException(e, sys)
