import os, sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_csv_file, read_yaml_file, read_csv_file, haversine
from feature_engine.outliers import Winsorizer


class FeatureEngineeringConfig:
    def __init__(self):
        self.fe_dir: str = os.path.join("artifacts", "feature_engineering")
        self.train_fe: str = os.path.join("artifacts", "feature_engineering", "train_fe.csv")
        self.test_fe: str = os.path.join("artifacts", "feature_engineering", "test_fe.csv")
        self.SCHEMA_PATH: str = os.path.join("config", "schema.yaml")

class FeatureEngineering:
    def __init__(self, config=FeatureEngineeringConfig()):
        try:
            self.config = config
            os.makedirs(self.config.fe_dir, exist_ok=True)
            self.schema = read_yaml_file(self.config.SCHEMA_PATH)
            self.drop_cols = self.schema.get("drop_cols", [])
            self.datetime_cols = self.schema.get("date_time_cols", [])
            self.strip_cols = self.schema.get("strip_cols", [])
            self.target_cols = self.schema.get("target_cols")

            logging.info(
                f"Schema loaded. Drop: {self.drop_cols}, Datetime: {self.datetime_cols}, "
                f"Strip: {self.strip_cols}, Target: {self.target_cols}"
            )
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------
    # 1. Basic Feature Improvement
    # ---------------------------
    def __feature_improvement(self, df: pd.DataFrame):
        try:
            logging.info("Feature improvement started")

            # Drop unnecessary columns
            order_date_col, time_ordered_col, time_picked_col = self.datetime_cols[:3]
            df.drop(self.drop_cols, axis=1, inplace=True, errors="ignore")

            # Clean text & handle NaNs
            df.replace("NaN ", np.nan, inplace=True)
            df.replace("conditions NaN", np.nan, inplace=True)
            df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)

            # Strip extra spaces
            for col in self.strip_cols:
                df[col] = df[col].str.strip()

            # Convert numeric fields
            df[['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']] = df[['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']].astype(float)

            # Clean & convert target
            df[self.target_cols] = df[self.target_cols].astype(str).str.extract('(\d+)')
            df[self.target_cols] = df[self.target_cols].astype(float)

            # Fill missing with mode and ffill
            df[time_ordered_col] = df[time_ordered_col].ffill().bfill()
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])


            logging.info("Feature improvement completed")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------
    # 2. Feature Construction
    # ---------------------------
    def __feature_construction(self, df: pd.DataFrame):
        try:
            logging.info("Feature construction started")

            # Dynamically get datetime column names
            order_date_col, time_ordered_col, time_picked_col = self.datetime_cols[:3]

            # Convert columns
            df[order_date_col] = pd.to_datetime(df[order_date_col], format="%d-%m-%Y")
            df[time_ordered_col] = pd.to_datetime(df[time_ordered_col], format='%H:%M:%S').dt.time
            df[time_picked_col] = pd.to_datetime(df[time_picked_col], format='%H:%M:%S').dt.time

            # Combine date and time
            df['order_datetime'] = df.apply(
                lambda row: pd.Timestamp.combine(row[order_date_col].date(), row[time_ordered_col])
                if pd.notnull(row[order_date_col]) and pd.notnull(row[time_ordered_col]) else pd.NaT,
                axis=1
            )

            df['pickup_datetime'] = df.apply(
                lambda row: pd.Timestamp.combine(row[order_date_col].date(), row[time_picked_col])
                if pd.notnull(row[order_date_col]) and pd.notnull(row[time_picked_col]) else pd.NaT,
                axis=1
            )

            # Drop original datetime columns
            df.drop([order_date_col, time_ordered_col, time_picked_col], axis=1, inplace=True, errors='ignore')

            # Derived temporal features
            df['prep_time(m)'] = (df['pickup_datetime'] - df['order_datetime']).dt.total_seconds() / 60
            df['order_hour'] = df['order_datetime'].dt.hour
            df['order_day_of_week'] = df['order_datetime'].dt.dayofweek
            df['is_weekend'] = df['order_day_of_week'].isin([5, 6]).astype(int)
            df['order_day'] = df['order_datetime'].dt.day
            df['order_week'] = df['order_datetime'].dt.isocalendar().week.astype(int)
            df['order_month'] = df['order_datetime'].dt.month

            # Distance features
            df['distance_km'] = haversine(
                df['Restaurant_latitude'], df['Restaurant_longitude'],
                df['Delivery_location_latitude'], df['Delivery_location_longitude']
            )

            df['manhattan_km'] = abs(df['Restaurant_latitude'] - df['Delivery_location_latitude']) + \
                                 abs(df['Restaurant_longitude'] - df['Delivery_location_longitude'])

            df['distance_per_speed'] = df['distance_km'] / (df['Vehicle_condition'] + 1e-5)
            df['distance_ratio'] = df['distance_km'] / (df['manhattan_km'] + 1e-5)
            df['rating_age_ratio'] = df['Delivery_person_Ratings'] / (df['Delivery_person_Age'] + 1e-5)
            df['rating_vehicle'] = df['Delivery_person_Ratings'] / (df['Vehicle_condition'] + 1e-5)

            # Drop temporary datetime columns
            df.drop(['order_datetime', 'pickup_datetime'], axis=1, inplace=True)
            logging.info("Feature construction completed successfully")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # 3. Main Pipeline
    def initiate_feature_engineering(self, train_path: str, test_path: str):
        try:
            logging.info("===== Feature Engineering Started =====")

            # Read data
            train_df = read_csv_file(train_path)
            test_df = read_csv_file(test_path)

            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # Apply steps
            train_df = self.__feature_improvement(train_df)
            test_df = self.__feature_improvement(test_df)

            train_df = self.__feature_construction(train_df)
            test_df = self.__feature_construction(test_df)

            # Save results
            save_csv_file(train_df, self.config.train_fe)
            save_csv_file(test_df, self.config.test_fe)
            logging.info("===== Feature Engineering Completed Successfully =====")
            return self.config.train_fe, self.config.test_fe

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        from src.components.data_ingestion import DataIngestion, DataIngestionConfig

        # Step 1: Run Data Ingestion
        ingestion_config = DataIngestionConfig()
        ingestion = DataIngestion(ingestion_config)
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed. Train: {train_path}, Test: {test_path}")

        # Step 2: Run Feature Engineering
        fe = FeatureEngineering()
        train_fe_path, test_fe_path = fe.initiate_feature_engineering(train_path, test_path)

        logging.info(f"Feature Engineering completed. Train FE: {train_fe_path}, Test FE: {test_fe_path}")
        print(f"✅ Feature Engineering Successful!\nTrain FE: {train_fe_path}\nTest FE: {test_fe_path}")
    except Exception as e:
        logging.error("❌ Error occurred during Feature Engineering pipeline.", exc_info=True)
        raise e
