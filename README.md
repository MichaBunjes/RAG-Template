# rag_template
## How tun run the backend Flask app:

1. Install requirements.txt
2. Create a `.env` file in the root directory with the following variables:
   ```
   AZURE_OPENAI_API_KEY=<your-api-key>
   AZURE_OPENAI_ENDPOINT=<your-endpoint>
   AZURE_EMBEDDING_MODEL_NAME=<your-embedding-model>
   AZURE_CHAT_MODEL_NAME=<your-chat-model>
   AZURE_OPENAI_API_VERSION=<your-api-version>
   ```
4. Prepare your PDF documents:
   - Create a folder called `pdf_data` in the root directory
   - Place your PDF files in subfolders within `pdf_data`. The subfolder names will be used as topics
   - Example structure:
     ```
     pdf_data/
     ├── topic1/
     │   ├── doc1.pdf
     │   └── doc2.pdf
     └── topic2/
         └── doc3.pdf
     ```

5. Run the Flask app:
   ```bash
   python main.py
   ```
   The app will run on http://localhost:8080 by default and will listen to HTTP POST requests with this JSON format:
   ```
    {
        "request_type": "rag_query",
        "user_question": "Hier deine Frage?",
        "messages": [
            {
            "text": "Hier deine Frage?",
            "isUser": true
            }
        ]
    }
    ```

6. To build your database either locally or in cloud, adjust and run pdf_to_index_pipeline.py
   - Extract text from your PDFs
   - Generate embeddings (this may take some time (hours))
   - Create a searchable database

7. Running main.py will start the backend server and the system will use the existing database.
   - Whether database is local or in cloud is specified in the class "RagSystem" in the main.py
   - If you modify the PDF files, you need to run the database generation again.
