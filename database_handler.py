import io
import logging
import os
from typing import Optional

import fitz
import pandas as pd
from google.cloud import storage

from model_communicator import ModelCommunicator


class DatabaseLoader:
    def __init__(
        self,
        bucket_name: Optional[str],
        file_path: str = "df_database.csv",
        is_database_in_cloud: bool = False,
    ) -> None:
        """Database loader class for managing CSV database files.

        This class provides functionality to load and retrieve database files from either local storage
        or Google Cloud Storage (GCS).

        Args:
            bucket_name (Optional[str]): Name of the GCS bucket. Required if accessing files in cloud storage.
            file_path (str, optional): Path to the database CSV file. Defaults to "df_database.csv".

        Attributes:
            file_path (str): Path to the database CSV file
            client (storage.Client): Google Cloud Storage client
            bucket_name (str): Name of the GCS bucket
            bucket (storage.Bucket): GCS bucket object
            blob (storage.Blob): GCS blob object for the database file
            logger (logging.Logger): Logger instance for this class
        """
        self.file_path = file_path
        self.is_database_in_cloud = is_database_in_cloud
        self.client = storage.Client()
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)
        self.blob = self.bucket.blob(self.file_path)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def load_df_csv_from_gcs(self) -> pd.DataFrame:
        """Load a CSV database file from Google Cloud Storage.

        Downloads a CSV file from the specified GCS bucket and loads it into a pandas DataFrame.

        Args:
            bucket_name (str): Name of the GCS bucket containing the CSV file
            file_path (str): Path to the CSV file within the bucket

        Returns:
            pd.DataFrame: DataFrame containing the loaded CSV data

        Note:
            This method requires that the GCS client and bucket have already been initialized
            in the class constructor.
        """
        content = self.blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(content))
        return df

    async def get_database(self) -> pd.DataFrame:
        """Get the database from either local storage or Google Cloud Storage.

        This method loads the database CSV file from either local storage or GCS depending on the
        is_in_cloud parameter.

        Args:
            is_in_cloud (bool, optional): Whether to load from cloud storage. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the loaded database, or None if loading fails

        Note:
            If loading from cloud storage (is_in_cloud=True), a bucket_name must have been provided
            during class initialization.
        """
        try:
            if self.is_database_in_cloud:
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


class DatabaseGenerator:
    def __init__(
        self,
        pdf_folder_path: str = "pdf_data",
        chunk_size: int = 500,
        overlap_size: int = 100,
    ) -> None:
        """Initialize a DatabaseGenerator instance.

        Creates a new DatabaseGenerator that processes PDF files into text chunks for database creation.

        Args:
            pdf_folder_path (str, optional): Path to folder containing PDF files. Defaults to "PDF Data".
            chunk_size (int, optional): Size of text chunks in characters. Defaults to 500.
            overlap_size (int, optional): Number of characters to overlap between chunks. Defaults to 100.

        Attributes:
            pdf_folder_path (str): Path to folder containing PDF files
            ModelCommunicator (ModelCommunicator): Instance of ModelCommunicator class
            chunk_size (int): Size of text chunks in characters
            overlap_size (int): Number of characters to overlap between chunks
            logger (logging.Logger): Logger instance for this class
        """
        self.pdf_folder_path = pdf_folder_path
        self.ModelCommunicator = ModelCommunicator()
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def embed_new_database(self, load_chunks_from_file: bool = False):
        if load_chunks_from_file:
            self.logger.info(
                "Importing PDf chunks..."
            )  # If chunks have been created already
            df_chunks = self.load_pdf_chunks()
        else:
            df_chunks = self.generate_pdf_chunks()

        # Embed new chunks
        self.logger.info(
            "Generating embeddings for database (this may take a long time)..."
        )
        df = ModelCommunicator.batch_generate_embedding_df(
            df_chunks, batch_size=10, sleep_time=5
        )
        return df

    def load_pdf_chunks(self, chunks_file_path: str = "df_pdf_chunks.csv"):
        """Load PDF chunks from a CSV file.

        Loads previously generated PDF chunks from a CSV file containing document text chunks
        and their metadata.

        Args:
            chunks_file_path (str, optional): Path to the CSV file containing the chunks.
                Defaults to "df_pdf_chunks.csv".

        Returns:
            pd.DataFrame: DataFrame containing the loaded chunks with columns:
                - topic: Topic/category of the document
                - document_name: Name of the PDF file
                - chunk_index: Index of the chunk within the document
                - page_number: Page number where the chunk starts
                - text: The actual text content of the chunk

            Returns None if loading fails.
        """
        try:
            self.logger.info("Importing PDF chunks...")
            df_chunks = pd.read_csv(chunks_file_path)
            return df_chunks
        except Exception:
            self.logger.error(
                f"Error loading PDF chunks df from file {chunks_file_path}: e"
            )
        return None

    def generate_pdf_chunks(self):
        """Generate text chunks from PDF files.

        Processes PDF files in the configured folder path, extracting text and splitting it into
        overlapping chunks. Each chunk includes metadata about its source document, location,
        and topic based on the folder structure.

        The method:
        1. Walks through the PDF folder structure
        2. Extracts text from each PDF file page by page
        3. Splits text into chunks of configured size with overlap
        4. Records metadata like topic, filename, chunk index and page number

        Returns:
            pd.DataFrame: DataFrame containing the generated chunks with columns:
                - topic: Topic/category based on folder structure
                - document_name: Name of the PDF file
                - chunk_index: Sequential index of chunk within document
                - page_number: PDF page number where chunk starts
                - text: The actual text content of the chunk

            Returns None if generation fails.
        """
        if not os.path.exists(self.pdf_folder_path):
            self.logger.error(f"PDF folder {self.pdf_folder_path} does not exist.")
            return None

        self.logger.info("Extracting PDF chunks from PDF files...")

        all_text_chunks = []

        for root, _, files in os.walk(self.pdf_folder_path):
            topic = os.path.relpath(root, self.pdf_folder_path)
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, filename)
                    try:
                        document = fitz.open(pdf_path)
                    except Exception as e:
                        self.logger.error(f"Error opening PDF file {pdf_path}: {e}")
                        continue

                    temp_text = ""
                    chunk_index = 0

                    for page_num in range(len(document)):
                        try:
                            page = document.load_page(page_num)

                            page_text = page.get_text()

                            temp_text += page_text

                            while len(temp_text) > self.chunk_size:
                                chunk = temp_text[: self.chunk_size]
                                chunk_data = {
                                    "topic": topic,
                                    "document_name": filename,
                                    "chunk_index": chunk_index,
                                    "page_number": page_num,
                                    "text": chunk,
                                }

                                all_text_chunks.append(chunk_data)
                                chunk_index += 1

                                # Update the temporary text variable to remove the overlap
                                temp_text = temp_text[
                                    self.chunk_size - self.overlap_size :
                                ]
                        except Exception as e:
                            self.logger.error(
                                f"Error processing page {page_num} in file {filename}: {e}"
                            )
                            continue

                    # Remaining text:
                    if temp_text:
                        chunk_data = {
                            "topic": topic,
                            "document_name": filename,
                            "chunk_index": chunk_index,
                            "page_number": page_num,
                            "text": temp_text,
                        }

                    all_text_chunks.append(chunk_data)

        return pd.DataFrame(all_text_chunks)


# def generate_new_database
# extract chunks from pdfs return df
# generate embeddings
# save_database
