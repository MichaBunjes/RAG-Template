# rag_template
## How tun run the backend Flask app:

1. Install requirements.txt
2. Create a `.env` file in the root directory with the following variables:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_VERSION=your_api_version 
   AZURE_OPENAI_ENDPOINT=your_endpoint
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
        "user_question": "Hier deine Frage?",
        "messages": [
            {
            "text": "Hier deine Frage?",
            "isUser": true
            }
        ]
    }
    ```

6. The first time you run the system, it will:
   - Extract text from your PDFs
   - Generate embeddings (this may take some time (hours))
   - Create a searchable database

7. For subsequent runs, the system will use the existing database. If you modify the PDF files, run the database generation again.
