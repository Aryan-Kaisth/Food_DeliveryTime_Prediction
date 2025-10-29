import os, sys
from dataclasses import dataclass
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_object, read_yaml_file

@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "model_trainer", "model.pkl")
    model_config_path: str = os.path.join("config", "model.yaml")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.model_file_path), exist_ok=True)
        logging.info(f"✅ ModelTrainer initialized. Model will be saved at: {self.config.model_file_path}")

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate model based on YAML configuration.
        """
        try:
            logging.info("🚀 Starting model training...")

            # --- Load model config ---
            model_cfg = read_yaml_file(self.config.model_config_path)
            model_name = model_cfg.get("model", "HistGradientBoostingRegressor")
            model_params = model_cfg.get("params", {})
            logging.info(f"📖 Loaded model config: {model_name} with params {model_params}")

            # --- Initialize model ---
            if model_name == "HistGradientBoostingRegressor":
                model = HistGradientBoostingRegressor(**model_params)
            else:
                raise ValueError(f"❌ Unsupported model type: {model_name}")

            logging.info(f"✅ Model initialized: {model_name}")

            # --- Train model ---
            model.fit(X_train, y_train)
            logging.info("✅ Model training completed")

            # --- Predictions ---
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # --- Metrics ---
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = root_mean_squared_error(y_train, y_train_pred)
            test_rmse = root_mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            logging.info(f"📊 Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
            logging.info(f"📉 Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            logging.info(f"📈 Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

            # --- Save trained model ---
            save_object(self.config.model_file_path, model)
            logging.info(f"💾 Trained model saved successfully at {self.config.model_file_path}")

            # --- Return summary ---
            return {
                "model_path": self.config.model_file_path,
                "model_name": model_name,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae
            }

        except Exception as e:
            logging.error("❌ Error during model training")
            raise CustomException(e, sys)


# ---- Testing ----
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion, DataIngestionConfig
    from src.components.data_transformation import DataTransformation

    # Step 1: Data ingestion
    ingest_config = DataIngestionConfig()
    data_ingestion = DataIngestion(config=ingest_config)
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Step 2: Data transformation (includes feature engineering)
    transformer = DataTransformation()
    X_train_transformed, X_test_transformed, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )

    # Step 3: Model training
    trainer = ModelTrainer()
    results = trainer.initiate_model_trainer(X_train_transformed, X_test_transformed, y_train, y_test)

    print("\n✅ Training Summary:")
    for k, v in results.items():
        print(f"{k}: {v}")
