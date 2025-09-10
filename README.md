# Multi-Provider RAG Chatbot

A Streamlit web application that implements a Retrieval-Augmented Generation (RAG) chatbot. This project was originally based on the final project for the *Generative AI Engineering with LLMs* certification by IBM. It allows users to upload a PDF document, select an LLM provider (Google, OpenAI, or Anthropic), and ask questions about the document's content.

**[➡️ View the Live Demo on Streamlit Cloud](https://ragmultichatbot.streamlit.app/)**

## Features

-   **Document Q&A:** Chat with your PDF documents.
-   **Multi-Provider Support:** Seamlessly switch between major LLM providers:
    -   Google (Gemini models)
    -   OpenAI (GPT models)
    -   Anthropic (Claude models)
-   **Conversational Memory:** The chatbot remembers the context of the current conversation to answer follow-up questions.
-   **Efficient Processing:** The processed document is cached, allowing for multiple conversations without reprocessing the PDF.
-   **Easy to Use:** Simple, clean interface powered by Streamlit.

## How it Works

The application follows a standard RAG pipeline:

1.  **Upload & Process:** A user uploads a PDF file.
2.  **Chunking:** The document is split into smaller, manageable text chunks using `RecursiveCharacterTextSplitter`.
3.  **Embedding & Indexing:** The text chunks are converted into vector embeddings (using a model from Google or OpenAI) and stored in an in-memory Chroma vector database.
4.  **Retrieval:** When a user asks a question, the app first formulates a "standalone" question based on chat history, then retrieves the most relevant chunks from the vector database.
5.  **Generation:** The original question, chat history, and the retrieved context are passed to the selected LLM to generate a comprehensive answer.

## Setup and Installation

### Prerequisites

-   Python 3.8+
-   API keys for the desired LLM providers (Google, OpenAI, Anthropic).

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

The primary way to use this application is through the public Streamlit Cloud deployment:

**https://ragmultichatbot.streamlit.app/**
 
1.  **Configure Provider:** Use the sidebar to select your desired LLM provider.
2.  **Enter API Key(s):** Enter the corresponding API key for the selected provider.
    -   *Note:* If you select Anthropic, you will be prompted for an OpenAI API key, as this application uses OpenAI's model for text embeddings.
3.  **Upload PDF:** Once the API key is entered, an uploader will appear. Upload the PDF document you want to chat with.
4.  **Ask Questions:** After the document is processed, the chat interface will appear. You can now ask questions about your document.
5.  **Start New Chat:** To upload a new document, click the "Start New Chat" button in the sidebar. This will clear the current document and conversation.

## Running Locally (for Development)

If you wish to run the application on your local machine for development, follow the "Setup and Installation" steps above, then run the following command:

```bash
streamlit run app.py
```

The application will be available in your browser at `http://localhost:8501`.

## Deployment Notes

This application includes a patch for `sqlite3` to use `pysqlite3-binary`. This is crucial for successful deployment on platforms like Streamlit Cloud, where the standard `sqlite3` library can be outdated. This patch is automatically applied at the very beginning of `app.py`.