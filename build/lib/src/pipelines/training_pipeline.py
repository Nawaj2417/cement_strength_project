import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    try:
        logging.info("Starting the training pipeline.")

        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data Transformation completed.")

        # Model Training
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)
        logging.info("Model Training completed.")

    except Exception as e:
        logging.error('Error occurred in training pipeline')
        raise CustomException(e, sys)
