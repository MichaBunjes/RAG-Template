import ast
import logging
import os
import tempfile
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
from google.cloud import storage


class IndexLoader:
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        file_path: str = "data/faiss_index.index",
        is_index_in_cloud: bool = False,
    ) -> None:
        """IndexHandler class for managing FAISS index operations.

        This class handles loading and searching the FAISS index for document chunks.
        It manages:
        - Loading index from local file or Google Cloud Storage
        - Performing similarity search on document vectors
        - Logging index operations

        Args:
            bucket_name (Optional[str]): Name of GCS bucket containing index file
            file_path (str, optional): Path to index file. Defaults to "data/faiss_index.csv".
            is_in_cloud (bool, optional): Whether index is stored in GCS. Defaults to False.

        Attributes:
            bucket_name (Optional[str]): Name of GCS bucket
            client (storage.Client): Google Cloud Storage client
            bucket (storage.Bucket): GCS bucket object
            blob (storage.Blob): GCS blob object for index file
            file_path (str): Path to index file
            is_in_cloud (bool): Whether index is in cloud storage
            index (faiss.Index): Loaded FAISS index
            logger (Logger): Logger instance for this class
        """
        self.is_in_cloud = is_index_in_cloud

        if is_index_in_cloud:
            self.bucket_name = bucket_name
            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
            self.blob = self.bucket.blob(file_path)

        self.file_path = file_path
        self.index = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def load_index_from_gcs(self) -> faiss.Index:
        """Load FAISS index from Google Cloud Storage.

        Downloads the index file from GCS to a temporary local file, loads it into a FAISS index,
        then removes the temporary file.

        Returns:
            faiss.Index: The loaded FAISS index object from cloud storage.
                Returns None if there is an error loading the index.

        Raises:
            Exception: If there is an error downloading or loading the index file
        """
        self.logger.info("Loading index from cloud storage")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            self.blob.download_to_filename(temp_path)
        index = faiss.read_index(temp_path)
        os.remove(temp_path)
        self.logger.info(f"FAISS index loaded from {self.file_path}")
        return index

    async def set_index(
        self,
    ) -> faiss.Index:
        """Set the FAISS index for similarity search.

        Loads the FAISS index either from Google Cloud Storage or local file system
        depending on is_in_cloud flag. Caches the loaded index in self.index.

        Returns:
            None

        Raises:
            ValueError: If bucket_name is not provided when is_in_cloud is True
            Exception: If there is an error loading the index file
        """
        try:
            if self.is_in_cloud:
                if not self.bucket_name:
                    raise ValueError(
                        "bucket_name must be provided when is_in_cloud is True"
                    )
                self.index = await self.load_index_from_gcs()
                return None

            else:
                self.index = faiss.read_index(self.file_path)
                self.logger.info(f"FAISS index loaded from {self.file_path}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting index: {e}")
            return None

    def similarity_search(self, query_vector, num_chunks: int = 30) -> List[int]:
        """Perform similarity search using the FAISS index.

        Takes a query vector and performs cosine similarity search against the indexed embeddings
        to find the most similar document chunks.

        Args:
            query_vector (numpy.ndarray): Query embedding vector to search against
            num_chunks (int, optional): Number of similar chunks to return. Defaults to 30.

        Returns:
            List[int]: List of indices for the most similar document chunks, sorted in ascending order.
                The indices correspond to rows in the original dataframe.

        Raises:
            Exception: If there is an error performing the similarity search
        """
        similarities, indices = self.index.search(query_vector, k=num_chunks)
        indices[0].sort()

        self.logger.info(
            f"{len(indices[0])} chunks found in documents (Cosine Similarity Search)."
        )
        return indices[0]


class IndexGenerator:
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        save_index_to_cloud: bool = False,
    ) -> None:
        self.save_index_to_cloud = save_index_to_cloud

        if save_index_to_cloud:
            self.bucket_name = bucket_name
            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)

        self.index = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def build_index(self, df: pd.DataFrame):
        """Build a FAISS index from document embeddings.

        Takes a DataFrame containing text embeddings and builds a FAISS index for similarity search.
        The embeddings must be in a 'text_embeddings' column as lists/arrays of floats.

        Args:
            df (pd.DataFrame): DataFrame containing text embeddings in 'text_embeddings' column

        Returns:
            faiss.IndexFlatIP: Built FAISS index containing the embeddings

        Raises:
            ValueError: If DataFrame does not contain 'text_embeddings' column
            ValueError: If embeddings are not in correct format (2D array)
            Exception: If there is an error building the index
        """
        if "text_embeddings" not in df.columns:
            self.logger.error("Dataframe does not contain 'text_embeddings' column.")
            raise ValueError("Dataframe does not contain 'text_embeddings' column.")

        if isinstance(df["text_embeddings"].iloc[0], str):
            self.logger.info("Converting string embeddings to float vectors...")
            df["text_embeddings"] = df["text_embeddings"].apply(
                lambda string: [float(x) for x in ast.literal_eval(string)]
            )
        embeddings = np.array(df["text_embeddings"].tolist())

        if embeddings.ndim != 2:
            self.logger.error("Embeddings must be a 2D array.")
            raise ValueError("Embeddings must be a 2D array.")

        d = embeddings.shape[1]
        self.logger.info(
            f"Building FAISS index with dimensionality {d} and Cosine Similarity Metric"
        )
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        self.logger.info(
            f"FAISS index built successfully with {index.ntotal} embeddings."
        )
        self.index = index
        self.logger.info("FAISS index built sucessfully.")

        return None

    def write_index(self, file_path: str = "data/faiss_index.index") -> None:
        if self.index is not None:
            if self.save_index_to_cloud:
                blob = self.bucket.blob(file_path)
                faiss.write_index(self.index, file_path)
                blob.upload_from_filename(file_path)
                os.remove(file_path)
                self.logger.info(
                    f"FAISS index uploaded to GCS, file path: {file_path}."
                )
            else:
                faiss.write_index(self.index, file_path)
                self.logger.info(f"FAISS index saved to {file_path}.")
        else:
            self.logger.error("FAISS index is None.")
