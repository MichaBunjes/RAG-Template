import io
import logging
from typing import Optional

import pandas as pd
from google.cloud import storage


class DatabaseHandler:
    def __init__(
        self, bucket_name: Optional[str], file_path: str = "df_database.csv"
    ) -> None:
        self.file_path = file_path
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)
        self.blob = self.bucket.blob(self.file_path)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def load_df_csv_from_gcs(self, bucket_name, file_path) -> pd.DataFrame:
        content = self.blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(content))
        return df

    async def get_database(
        self,
        is_in_cloud: bool = False,
    ) -> pd.DataFrame:
        try:
            if is_in_cloud:
                if not self.bucket_name:
                    raise ValueError(
                        "bucket_name must be provided when is_in_cloud is True"
                    )
                df = await self.load_df_csv_from_gcs(self.bucket_name, self.file_path)
                return df
            else:
                return pd.read_csv(self.file_path)
        except Exception as e:
            self.logger.error(f"Error getting database: {e}")
            return None


# class DatabaseGenerator:
# self file_path
# self chunksize
# self overlap size
# self directory path
# self ModelCommunicator

# def generate_new_database
# extract chunks from pdfs return df
# generate embeddings
# save_database
