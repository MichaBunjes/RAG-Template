import asyncio
import logging
from typing import List, Tuple

import numpy as np

from database_handler import DatabaseLoader
from index_handler import IndexHandler
from model_communicator import ModelCommunicator


class QueryHandler:
    def __init__(self, index, df) -> None:
        """QueryHandler class for managing conversational queries with RAG.

        This class coordinates between the database, index, and model components to handle
        user queries and generate contextual responses. It manages:
        - Loading and accessing the vector database and FAISS index
        - Converting message history into conversation format
        - Retrieving relevant context for queries
        - Generating responses using the language model

        Attributes:
            DatabaseLoader: Handler for accessing the document database
            IndexHandler: Handler for similarity search using FAISS index
            ModelCommunicator: Interface for LLM and embedding model interactions
            index: Loaded FAISS index for similarity search
            df: Loaded document database as DataFrame
            logger: Logger instance for this class
        """
        if index is None or df is None:
            raise RuntimeError(
                "Use 'await QueryHandler.create()' instead of 'QueryHandler()'."
            )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.DatabaseLoader = DatabaseLoader(is_database_in_cloud=False)
        self.IndexHandler = IndexHandler(is_index_in_cloud=False)
        self.ModelCommunicator = ModelCommunicator(
            embedding_model="placeholder_embedding_model",
            chat_model="placeholder_chat_model",
        )

        self.index = index
        self.df = df

    @classmethod
    async def create(cls):
        """Factory method to create a new QueryHandler instance.

        This class method asynchronously initializes a QueryHandler by loading the required
        FAISS index and document database. It uses asyncio.gather to load both resources
        concurrently for better performance.

        Returns:
            QueryHandler: A new QueryHandler instance with loaded index and database

        Note:
            This method should be used instead of directly instantiating QueryHandler
            since it handles the asynchronous loading of required resources.
        """
        index, df = await asyncio.gather(
            cls.IndexHandler.get_index(), cls.DatabaseLoader.get_database()
        )
        return cls(index, df)

    def convert_messages_to_conversation_history(self, messages: List[dict]):
        """Convert message history into conversation format.

        Takes a list of message dictionaries containing isUser flag and text content,
        and converts them into a list of conversation entries with role and content.

        Args:
            messages (List[dict]): List of message dictionaries with keys:
                - isUser (bool): Flag indicating if message is from user
                - text (str): Message content

        Returns:
            List[dict]: List of conversation entries with keys:
                - role (str): Either "user" or "assistant"
                - content (str): Message content

        Example:
            >>> messages = [
                    {"isUser": True, "text": "Hello"},
                    {"isUser": False, "text": "Hi there"}
                ]
            >>> convert_messages_to_conversation_history(messages)
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        """
        conversation_history = []
        for message in messages:
            role = "user" if message["isUser"] else "assistant"
            conversation_history.append({"role": role, "content": message["text"]})
        return conversation_history

    async def handle_query(
        self, user_question: str, messages: List[dict] = None
    ) -> str:
        """Handle an incoming user query and generate a response.

        This method processes a user question and optional message history to generate a contextually
        relevant response. It performs the following steps:
        1. Processes conversation history if provided
        2. Combines recent user questions for context retrieval
        3. Generates embeddings for the combined questions
        4. Performs similarity search to find relevant context
        5. Constructs prompts and generates LLM response

        Args:
            user_question (str): The current question from the user
            messages (List[dict], optional): List of previous conversation messages. Each message
                should contain:
                - isUser (bool): Flag indicating if message is from user
                - text (str): Message content

        Returns:
            str: The generated response from the LLM

        Raises:
            Exception: If there is an error during query processing
        """
        try:
            if messages:
                conversation_history = self.convert_messages_to_conversation_history(
                    messages
                )

            self.logger.info("Handling user query...")

            # Combine queries for context retrieval
            last_questions = (
                2  # The last 2 user question (including the current) will be used
            )
            combined_user_questions = user_question

            if conversation_history:
                user_entries = [
                    entry["content"]
                    for entry in conversation_history
                    if entry["role"] == "user"
                ]
                # assistant_entries = [
                #     entry["content"]
                #     for entry in conversation_history
                #     if entry["role"] == "assistant"
                # ]
                combined_user_questions += "\n".join(user_entries[-last_questions:])
                # combined_user_questions += "\n".join(assistant_entries[-2:])

            # Generate Embedding
            query_embedding = await np.array(
                self.ModelCommunicator.generate_embeddings_from_list(
                    [combined_user_questions]
                )
            )

            # Similarity Search
            indices_similar = IndexHandler.similarity_search(
                query_embedding, num_chunks=30
            )

            # Generate prompt
            system_prompt, context_prompt, query_prompt = self.get_prompts(
                indices_similar, user_question
            )
            llm_input_messages = []
            llm_input_messages = [
                {"role": "system", "content": system_prompt + context_prompt}
            ]

            # Add conversation history to input messages
            if conversation_history:
                for entry in conversation_history:
                    llm_input_messages.append(entry)

            # Add User query to input messages
            llm_input_messages.append({"role": "user", "content": query_prompt})

            # Generate LLM response
            llm_response = self.ModelCommunicator.generate_chat_response(
                llm_input_messages
            )

            # Process LLM response
            # TODO: make document links

            return llm_response
        except Exception as e:
            self.logger.error(f"Error handling query: {e}")
            return "Ich bin leider nicht in der Lage, doie Anfrage zu beantworten. Bitte versuche es erneut."

    def get_prompts(
        self, indices_similar, user_question: str = "Ich habe keine Frage."
    ) -> Tuple[str, str, str]:
        """Get prompts for the LLM query.

        Generates system, context and query prompts for the LLM by combining the user question
        with relevant context from similar document chunks.

        Args:
            indices_similar (List[int]): List of indices for similar chunks from similarity search
            user_question (str, optional): User's question. Defaults to "Ich habe keine Frage."

        Returns:
            Tuple[str, str, str]: A tuple containing:
                - system_prompt: Instructions for the LLM's role
                - context_prompt: Context from relevant document chunks
                - query_prompt: The formatted user question
        """
        system_prompt = "Du bist ein hilfreicher RAG-Chatbot. Dine detaillierten Antworten basieren ausschließlich auf den folgenden Kontextausschnitten."
        query_prompt = f"**Anfrage**: {user_question}"
        context_prompt = "\n\n**Kontextausschnitte:**\n"
        for i, row in self.df.iloc[indices_similar].iterrows():
            context_prompt += f"**Dokument: {row['document_name']}, Seite {row['page_number']}**: \n{row['text']}\n\n"

        return system_prompt, context_prompt, query_prompt
