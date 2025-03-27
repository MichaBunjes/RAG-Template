import logging
import os
import time
from typing import List

import pandas as pd
import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI


class ModelCommunicator:
    def __init__(self) -> None:
        """ModelCommunicator class for interacting with language models.

        This class handles communication with Azure OpenAI models for embeddings and chat.
        It manages:
        - Authentication with Azure OpenAI API
        - Generating embeddings from text using the embedding model
        - Generating chat responses using the chat model
        - Batch processing of embeddings for large datasets

        Args:
            embedding_model (str): Name of the embedding model to use
            chat_model (str): Name of the chat model to use

        Attributes:
            api_key (str): Azure OpenAI API key from environment variables
            api_version (str): Azure OpenAI API version from environment variables
            api_endpoint (str): Azure OpenAI endpoint URL from environment variables
            client (AzureOpenAI): Authenticated Azure OpenAI client
            embedding_model (str): Name of embedding model being used
            chat_model (str): Name of chat model being used
            temperature (float): Temperature parameter for chat generation
            dimensions (int): Dimension size of generated embeddings
            logger (Logger): Logger instance for this class
        """
        load_dotenv()
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.embedding_model = os.getenv("AZURE_EMBEDDING_MODEL_NAME")
        self.chat_model = os.getenv("AZURE_CHAT_MODEL_NAME")

        if not self.api_key or not self.api_version or not self.api_endpoint:
            raise ValueError("API credentials are not set in environment variables.")

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_endpoint,
        )
        self.temperature = 0.0
        self.dimensions = 768

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_embeddings_from_list(self, chunk_list: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text chunks.

        Takes a list of text chunks and generates embeddings using the Azure OpenAI embedding model.
        The embeddings are used for similarity search and retrieval.

        Args:
            chunk_list (List[str]): List of text chunks to generate embeddings for

        Returns:
            List[List[float]]: List of embeddings, where each embedding is a list of floats.
                Returns None if there is an error generating embeddings.

        Raises:
            Exception: If there is an error calling the embeddings API
        """
        try:
            embeddings_response = self.client.embeddings.create(
                input=chunk_list, model=self.embedding_model, dimensions=self.dimensions
            )
            return [embedding.embedding for embedding in embeddings_response.data]
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return None

    def batch_generate_embedding_df(
        self, df: pd.DataFrame, batch_size: int = 10, sleep_time: int = 5
    ) -> pd.DataFrame:
        """Batch generate embeddings for text chunks in a DataFrame.

        Takes a DataFrame containing text chunks and generates embeddings in batches to avoid
        rate limits. Adds a new column 'text_embeddings' with the generated embeddings.

        Args:
            df (pd.DataFrame): DataFrame containing a 'text' column with text chunks to embed
            batch_size (int, optional): Number of chunks to process in each batch. Defaults to 10.
            sleep_time (int, optional): Seconds to sleep between batches. Defaults to 5.

        Returns:
            pd.DataFrame: Input DataFrame with new 'text_embeddings' column containing embeddings

        Raises:
            ValueError: If DataFrame does not contain required 'text' column
            Exception: If there is an error generating embeddings
        """
        chunk_list = df["text"].tolist()
        all_embeddings = []

        self.logger.info("Generating batch embeddings...")
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
        """Generate a chat response from the language model.

        Takes a list of message dictionaries containing role and content and generates
        a response using the configured chat model.

        Args:
            messages (List[dict]): List of message dictionaries with keys:
                - role (str): The role of the message sender ("system", "user", or "assistant")
                - content (str): The message content

        Returns:
            str: The generated response from the language model.
                Returns error message if there is a problem generating the response.

        Raises:
            Exception: If there is an error calling the chat completions API
        """
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
