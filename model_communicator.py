import logging
import os
import time
from typing import List

import pandas as pd
import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI


class ModelCommunicator:
    def __init__(self, embedding_model, chat_model) -> None:
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_VERSION")
        self.api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not self.api_key or not self.api_version or not self.api_endpoint:
            raise ValueError("API credentials are not set in environment variables.")

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_endpoint,
        )
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.temperature = 0.0
        self.dimensions = 768

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def generate_embeddings_from_list(
        self, chunk_list: List[str]
    ) -> List[List[float]]:
        """Embeds a list of text chunks using the embedding model"""
        try:
            embeddings_response = await self.client.embeddings.create(
                input=chunk_list, model=self.embedding_model, dimensions=self.dimensions
            )
            return [embedding.embedding for embedding in embeddings_response.data]
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return None

    def batch_generate_embedding_df(
        self, df: pd.DataFrame, batch_size: int = 10, sleep_time: int = 5
    ) -> pd.DataFrame:
        """Takes a dataframe with a column 'text' containing all text chunks to embed and adds a new column with the corresponding embeddings"""
        chunk_list = df["text"].tolist()
        all_embeddings = []

        self.logger.info("Generateing batch embeddings...")
        for i in tqdm.tqdm(
            range(0, len(chunk_list), batch_size), desc="Generating embeddings"
        ):
            batch = chunk_list[i : i + batch_size]
            embeddings = self.generate_embeddings_from_list(batch)
            all_embeddings.extend(embeddings)
            time.sleep(sleep_time)

        df["text_embeddings"] = all_embeddings
        return df

    def generate_chat_response(self, messages: List[dict]) -> str:
        self.logger.info(
            f"Generating LLM response (temperature: {self.temperature})..."
        )
        try:
            llm_response = self.client.chat.completions.create(
                model=self.chat_model, temperature=self.temperature, messages=messages
            )
            return llm_response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {e}")
            return "I am not able to generate a response. Please try again."
