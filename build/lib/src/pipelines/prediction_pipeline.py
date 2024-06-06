import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self) :
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Cement: float, Blast_Furnace_Slag: float, Fly_Ash: float, Water: float,
                 Superplasticizer: float, Coarse_Aggregate: float, Fine_Aggregate: float, Age_day: float):
        self.Cement = Cement
        self.Blast_Furnace_Slag = Blast_Furnace_Slag
        self.Fly_Ash = Fly_Ash
        self.Water = Water
        self.Superplasticizer = Superplasticizer
        self.Coarse_Aggregate = Coarse_Aggregate
        self.Fine_Aggregate = Fine_Aggregate
        self.Age_day = Age_day

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Cement': [self.Cement],
                'Blast Furnace Slag': [self.Blast_Furnace_Slag],
                'Fly Ash': [self.Fly_Ash],
                'Water': [self.Water],
                'Superplasticizer': [self.Superplasticizer],
                'Coarse Aggregate': [self.Coarse_Aggregate],
                'Fine Aggregate': [self.Fine_Aggregate],
                'Age (day)': [self.Age_day]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            logging.info("Exception occurred in get_data_as_dataframe")
            raise CustomException(e, sys)
