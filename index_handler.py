import ast
import logging
import os
import tempfile
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
from google.cloud import storage


class IndexHandler:
    def __init__(
        self,
        bucket_name: Optional[str],
        file_path: str = "faiss_index.csv",
        is_in_cloud: bool = False,
    ) -> None:
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.blob = self.bucket.blob(file_path)
        self.file_path = file_path
        self.is_in_cloud = is_in_cloud
        self.index = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def load_index_from_gcs(self):
        self.logger.info("Loading index from cloud storage")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            await self.blob.download_to_filename(temp_path)
        index = faiss.read_index(temp_path)
        os.remove(temp_path)
        self.logger.info(f"FAISS index loaded from {self.file_path}")
        return index

    async def get_index(
        self,
    ) -> faiss.Index:
        try:
            if self.is_in_cloud:
                if not self.bucket_name:
                    raise ValueError(
                        "bucket_name must be provided when is_in_cloud is True"
                    )
                index = await self.load_index_from_gcs()
                return index

            else:
                index = await faiss.read_index(self.file_path)
                self.logger.info(f"FAISS index loaded from {self.file_path}")
                return index

        except Exception as e:
            self.logger.error(f"Error getting index: {e}")
            return None

    def similarity_search(self, query_vector, num_chunks: int = 30) -> List[int]:
        similarities, indices = self.index.search(query_vector, k=num_chunks)
        indices[0].sort()

        self.logger.info(
            f"{len(indices[0])} chunks found in documents (Cosine Similarity Search)."
        )
        return indices[0]

    async def build_index(self, df: pd.DataFrame) -> faiss.IndexFlatIP:
        if "text_embeddings" not in df.colums:
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
        self.logger.info(f"Building FAISS index with dimensionality {d}.")
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        self.logger.info(
            f"FAISS index built successfully with {index.ntotal} embeddings."
        )
        return index

    async def save_index_as_file(self, index) -> None:
        faiss.write_index(index, self.file_path)
        self.logger.info(f"FAISS index saved to {self.file_path}.")
