import os, sys
from src.logger import logging
from src.exception import CustomException

def run_training_pipeline():
    """
    Runs the full training pipeline:
    1. Data ingestion
    2. Data transformation
    3. Model training
    """

    try:
        logging.info("===== Starting Training Pipeline =====")

        # --- Data Ingestion ---
        from src.components.data_ingestion import DataIngestion, DataIngestionConfig
        ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(config=ingestion_config)
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train: {train_path}, Test: {test_path}")

        # --- Data Transformation ---
        from src.components.data_transformation import DataTransformation
        transformer = DataTransformation()
        X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
            train_path=train_path,
            test_path=test_path
        )
        logging.info(f"Data transformation completed. X_train: {X_train.shape}, X_test: {X_test.shape}")

        # --- Model Training ---
        from src.components.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        model = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
        logging.info("Model training and evaluation completed successfully!")

        logging.info("===== Training Pipeline Completed =====")
        return model

    except Exception as e:
        logging.error("Error during training pipeline")
        raise CustomException(e, sys)


