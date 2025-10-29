import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from feature_engine.outliers import Winsorizer
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import (
    save_numpy_array_data, save_object, read_csv_file, read_yaml_file
)
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringConfig


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
    transformed_train_file_path: str = os.path.join("artifacts", "data_transformation", "train.npy")
    transformed_test_file_path: str = os.path.join("artifacts", "data_transformation", "test.npy")
    SCHEMA_PATH = os.path.join("config", "schema.yaml")


class DataTransformation:
    def __init__(self):
        try:
            self.config = DataTransformationConfig()
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)

            # Load schema
            self.schema = read_yaml_file(self.config.SCHEMA_PATH)
            self.ohe_cols = self.schema.get("ohe_cols", [])
            self.ordinal_cols = self.schema.get("ordinal", [])
            self.target_cols = self.schema.get("target_cols")
            self.winsor_cols = self.schema.get("winsor_cols", [])

            logging.info(
                f"Schema loaded successfully. winsor_cols: {self.winsor_cols}, "
                f"Ordinal: {self.ordinal_cols}, OHE: {self.ohe_cols}, Target: {self.target_cols}"
            )

        except Exception as e:
            logging.error("Error initializing DataTransformation")
            raise CustomException(e, sys)

    def __get_preprocessor_pipeline(self):
        try:
            traffic_order = ['Low', 'Medium', 'High', 'Jam']

            preprocessor = ColumnTransformer(transformers=[
                ("ordinal", OrdinalEncoder(categories=[traffic_order]), self.ordinal_cols),
                ("ohe", OneHotEncoder(drop="first", sparse_output=True), self.ohe_cols)
            ], remainder="passthrough")

            final_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("scaler", StandardScaler())
            ])

            return final_pipeline
        except Exception as e:
            logging.error("Error creating preprocessing pipeline")
            raise CustomException(e, sys)

    def __handle_outliers(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Fit winsorizer on train only, then transform both train and test.
        """
        try:
            logging.info("Starting Winsorization on numeric columns")

            winsor = Winsorizer(
                capping_method='iqr',
                tail='both',
                fold=1.5,
                variables=list(self.winsor_cols)
            )

            X_train[self.winsor_cols] = winsor.fit_transform(X_train[self.winsor_cols])
            X_test[self.winsor_cols] = winsor.transform(X_test[self.winsor_cols])

            logging.info("Outlier handling done successfully (fit on train, transform on test)")
            return X_train, X_test

        except Exception as e:
            logging.error("Error handling outliers")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("===== Data Transformation Started =====")

            # 1️⃣ Perform Feature Engineering
            fe = FeatureEngineering()
            logging.info("Running Feature Engineering Pipeline...")
            train_fe_path, test_fe_path = fe.initiate_feature_engineering(train_path, test_path)

            # 2️⃣ Read processed feature-engineered data
            train_df = read_csv_file(train_fe_path)
            test_df = read_csv_file(test_fe_path)

            logging.info(f"Feature engineered train shape: {train_df.shape}, test shape: {test_df.shape}")

            # 3️⃣ Split features and target
            X_train = train_df.drop(columns=[self.target_cols], axis=1)
            y_train = train_df[self.target_cols]
            X_test = test_df.drop(columns=[self.target_cols], axis=1)
            y_test = test_df[self.target_cols]

            # 4️⃣ Handle Outliers
            X_train, X_test = self.__handle_outliers(X_train, X_test)

            # 5️⃣ Create preprocessing pipeline
            preprocessor = self.__get_preprocessor_pipeline()

            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # 6️⃣ Save transformed arrays and preprocessor
            save_numpy_array_data(self.config.transformed_train_file_path, np.c_[X_train_transformed, y_train])
            save_numpy_array_data(self.config.transformed_test_file_path, np.c_[X_test_transformed, y_test])
            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            logging.info("===== Data Transformation Completed Successfully =====")

            return X_train_transformed, X_test_transformed, y_train, y_test

        except Exception as e:
            logging.error("Error in data transformation pipeline")
            raise CustomException(e, sys)


# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig
    from src.components.data_transformation import DataTransformation  # make sure it's imported

    # Initialize data ingestion
    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)

    # Perform ingestion to get train and test file paths
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Initialize DataTransformation
    transformer = DataTransformation()

    # Run the data transformation
    X_train_transformed, X_test_transformed, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )

    # Quick sanity checks
    print("✅ Transformed X_train shape:", X_train_transformed.shape)
    print("✅ Transformed X_test shape:", X_test_transformed.shape)
    print("✅ y_train shape:", y_train.shape)
    print("✅ y_test shape:", y_test.shape)
