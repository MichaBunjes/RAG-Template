import logging

from database_handler import DatabaseGenerator
from index_handler import IndexGenerator


class pdf_to_index_pipeline:
    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        return None

    def run_pipeline(self):
        # PDF Chunks
        database_generator = DatabaseGenerator(
            pdf_folder_path="pdf_data",
            chunk_size=500,
            overlap_size=100,
            save_pdf_chunks_to_cloud=False,
        )
        df_chunks, df_chunks_and_embeddings = database_generator.embed_new_database(
            load_chunks_from_file=False
        )
        database_generator.write_pdf_chunks_database_to_file(
            df_chunks, file_path="df_database.parquet"
        )

        # Index
        index_generator = IndexGenerator(save_index_to_cloud=False)
        index_generator.build_index(df_chunks_and_embeddings)
        index_generator.write_index(file_path="faiss_index.index")


pipeline = pdf_to_index_pipeline()
pipeline.run_pipeline()
