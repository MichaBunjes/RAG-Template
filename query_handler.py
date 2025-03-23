import asyncio
import logging
from typing import List

import numpy as np

from database_handler import DatabaseHandler
from index_handler import IndexHandler
from model_communicator import ModelCommunicator


class QueryHandler:
    async def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.DatabaseHandler = DatabaseHandler()  # Loads database
        self.IndexHandler = IndexHandler()  # Loads Index
        self.ModelCommunicator = ModelCommunicator(
            embedding_model="placeholder_embedding_model",
            chat_model="placeholder_chat_model",
        )
        self.index, self.df = await asyncio.gather(
            self.IndexHandler.get_index(), self.DatabaseHandler.get_database()
        )

    def convert_messages_to_conversation_history(self, messages: List[dict]):
        conversation_history = []
        for message in messages:
            role = "user" if message["isUser"] else "assistant"
            conversation_history.append({"role": role, "content": message["text"]})
        return conversation_history

    async def handle_query(
        self, user_question: str, messages: List[dict] = None
    ) -> str:
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
    ):
        system_prompt = "Du bist ein hilfreicher RAG-Chatbot. Dine detaillierten Antworten basieren ausschließlich auf den folgenden Kontextausschnitten."
        query_prompt = f"**Anfrage**: {user_question}"
        context_prompt = "\n\n**Kontextausschnitte:**\n"
        for i, row in self.df.iloc[indices_similar].iterrows():
            context_prompt += f"**Dokument: {row['document_name']}, Seite {row['page_number']}**: \n{row['text']}\n\n"

        return system_prompt, context_prompt, query_prompt
