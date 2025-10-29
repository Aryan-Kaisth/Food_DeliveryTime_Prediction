import os, sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import load_object


class PredictionPipeline:
    def __init__(self):
        """
        Initializes the prediction pipeline by loading the preprocessor and trained model.
        """
        try:
            preprocessor_path = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model_trainer", "model.pkl")

            # Load preprocessor and model
            self.preprocessor = load_object(preprocessor_path)
            self.model = load_object(model_path)

            logging.info("✅ PredictionPipeline initialized successfully.")
        except Exception as e:
            logging.error("❌ Error initializing PredictionPipeline.")
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Transforms input features using the preprocessor and generates model predictions.
        Args:
            features (pd.DataFrame): Raw feature DataFrame (same schema as training data).
        Returns:
            np.ndarray: Model predictions.
        """
        try:

            logging.info("🔄 Transforming input features using preprocessor...")
            transformed_features = self.preprocessor.transform(features)

            logging.info("🧠 Generating predictions using trained model...")
            preds = self.model.predict(transformed_features)

            logging.info(f"✅ Predictions generated successfully. Shape: {preds.shape}")
            return preds

        except Exception as e:
            logging.error("❌ Error occurred during prediction.")
            raise CustomException(e, sys)