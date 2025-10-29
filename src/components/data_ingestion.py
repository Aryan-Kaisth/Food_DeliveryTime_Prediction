# data_ingestion.py
import os, sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import save_csv_file, read_yaml_file
from db.connection import get_connection
from db.queries import get_all_data


@dataclass
class DataIngestionConfig:
    """Holds file paths and parameters for the data ingestion process."""
    raw_data_dir: str = os.path.join("artifacts", "data_ingestion")
    raw_data_path: str = os.path.join("artifacts", "data_ingestion", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")
    tables_path: str = os.path.join("config", "db_tables.yaml")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        """Initialize with a configuration object."""
        self._config = config
        os.makedirs(self._config.raw_data_dir, exist_ok=True)

        # Load table names from config
        self._tables = read_yaml_file(self._config.tables_path)
        self._data_table = self._tables.get("data_table")

    def __fetch_data_from_db(self, table_name: str) -> pd.DataFrame:
        """Fetch all rows from a DB table as a pandas DataFrame (private)."""
        try:
            with get_connection() as conn:
                stmt = get_all_data(table_name)
                result = conn.execute(stmt)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                logging.info(f"Fetched {df.shape[0]} rows from table '{table_name}'")
                return df
        except Exception as e:
            logging.error(f"Failed to fetch data from table '{table_name}'")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        """Public method: performs the entire data ingestion process."""
        logging.info("===== Data Ingestion Process Started =====")
        try:
            # Fetch Data from DB
            data = self.__fetch_data_from_db(self._data_table)
            logging.info(f"Data shape: {data.shape}")

            # Save raw copy
            save_csv_file(data, self._config.raw_data_path)
            logging.info(f"Raw data saved at {self._config.raw_data_path}")

            # Split train/test
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            # Save train/test sets
            save_csv_file(train_set, self._config.train_data_path)
            save_csv_file(test_set, self._config.test_data_path)

            logging.info("===== Data Ingestion Completed Successfully =====")
            return self._config.train_data_path, self._config.test_data_path

        except Exception as e:
            logging.error("Error occurred during data ingestion.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
