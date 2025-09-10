# Patches sqlite3 to use pysqlite3-binary.
# This is required for ChromaDB to work correctly in environments like Streamlit Cloud.
# See: https://docs.trychroma.com/troubleshooting#sqlite
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Standard library imports
import os
import tempfile
from typing import List

# Third-party imports
import streamlit as st
import nest_asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Apply nest_asyncio to allow running asyncio event loops within other event loops,
# which is common in environments like Jupyter or Streamlit.
nest_asyncio.apply()

# --- Application Configuration ---

PROVIDER_MAP = {
    "Google": {
        "llm_class": ChatGoogleGenerativeAI,
        "embedding_class": GoogleGenerativeAIEmbeddings,
        "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
        "embedding_model_name": "models/embedding-001",
        "embedding_provider": "Google",
    },
    "OpenAI": {
        "llm_class": ChatOpenAI,
        "embedding_class": OpenAIEmbeddings,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "embedding_model_name": None,
        "embedding_provider": "OpenAI",
    },
    "Anthropic": {
        "llm_class": ChatAnthropic,
        "embedding_class": OpenAIEmbeddings,  # Fallback to OpenAI for embeddings
        "models": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        "embedding_model_name": None,
        "embedding_provider": "OpenAI",  # Explicitly use OpenAI for embeddings
    }
}

# --- Core Logic Functions ---

def get_llm(provider: str, model: str, api_key: str):
    """Initializes and returns a language model instance based on the provider."""
    config = PROVIDER_MAP.get(provider)
    if not config:
        return None
    
    api_key_name = f"{provider.lower()}_api_key"
    return config["llm_class"](
        model=model, 
        temperature=0.5, 
        **{api_key_name: api_key}
    )

def get_embedding_model(provider: str, api_key: str):
    """Initializes and returns an embedding model instance."""
    config = PROVIDER_MAP.get(provider)
    if not config:
        return None

    embedding_class = config["embedding_class"]
    
    if provider == "Google":
        return embedding_class(model=config["embedding_model_name"], google_api_key=api_key)
    elif provider == "OpenAI":
        # This also covers the Anthropic case which uses OpenAI embeddings
        return embedding_class(openai_api_key=api_key)
    
    return None

@st.cache_resource(show_spinner="Processing PDF...")
def create_retriever_from_pdf(file_contents: bytes, embedding_provider: str, embedding_api_key: str):
    """
    Creates a retriever from the contents of a PDF file.
    The function saves the file temporarily, loads and splits it into chunks,
    creates embeddings, and stores them in a Chroma vector store.
    The retriever is cached to avoid reprocessing the same file.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(file_contents)
            temp_filepath = tf.name

        loader = PyPDFLoader(temp_filepath)
        documents = loader.load()
        os.unlink(temp_filepath) # Use unlink for consistency with tempfile
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        embedding_model = get_embedding_model(embedding_provider, embedding_api_key)
        if embedding_model is None:
            st.error("Could not create embedding model. Check provider configuration.")
            return None

        vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_model)
        return vectordb.as_retriever()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# --- Session State and History Management ---

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        "messages": [],
        "store": {},
        "retriever": None,
        "file_key": None,
        "file_name": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_chat_session():
    """Resets the chat session and clears related session state variables."""
    initialize_session_state() # Ensure all keys exist before resetting
    st.session_state.messages = []
    st.session_state.store = {}
    st.session_state.retriever = None
    st.session_state.file_key = None
    st.session_state.file_name = None
    st.rerun()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves the chat history for a given session ID.
    If no history exists, a new one is created.
    """
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def format_docs(docs: List[Document]) -> str:
    """Formats a list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- Main Application UI and Logic ---

st.title("📄 Multi-Provider RAG Q&A")
st.markdown("Upload a PDF document, choose your provider, and ask any question.")

initialize_session_state()

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    selected_provider = st.selectbox(
        "Choose your LLM provider:",
        PROVIDER_MAP.keys()
    )

    api_key = st.text_input(f"Enter your {selected_provider} API Key:", type="password")
    
    provider_models = PROVIDER_MAP[selected_provider]["models"]
    selected_model = st.selectbox("Choose your model:", provider_models)

    # Determine which provider and API key to use for embeddings
    embedding_provider = PROVIDER_MAP[selected_provider]["embedding_provider"]
    embedding_api_key = api_key
    if embedding_provider != selected_provider:
        st.info(f"{selected_provider} requires an {embedding_provider} API key for embeddings.")
        embedding_api_key = st.text_input(
            f"Enter {embedding_provider} API Key (for embeddings):", 
            type="password", 
            key="embedding_key"
        )

    st.markdown("---")
    if st.session_state.file_name:
        st.info(f"**Active File:** `{st.session_state.file_name}`")
        if st.button("Start New Chat"):
            reset_chat_session()

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main application flow
if not api_key or (embedding_provider != selected_provider and not embedding_api_key):
    st.warning(f"Please enter your {selected_provider} API Key in the sidebar to begin.")
elif not st.session_state.retriever:
    st.info("Please upload a PDF file to begin.")
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    
    if uploaded_file:
        # Clear previous session and process the new file
        st.session_state.messages = []
        st.session_state.store = {}
        
        st.session_state.retriever = create_retriever_from_pdf(
            uploaded_file.getvalue(), 
            embedding_provider, 
            embedding_api_key
        )
        # Create a unique key for the file to manage session history
        st.session_state.file_key = f"{uploaded_file.name}-{uploaded_file.size}"
        st.session_state.file_name = uploaded_file.name
        st.rerun()
else:
    # --- RAG Chain and Chat Interface ---
    llm = get_llm(selected_provider, selected_model, api_key)
    
    # 1. Contextualizer: Reformulates the user's question based on chat history
    # to create a standalone question for the retriever.
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = (
        contextualize_q_prompt | llm | StrOutputParser() | st.session_state.retriever
    )
    
    # 2. QA Prompt: The main prompt for answering the question based on retrieved context.
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # 3. Question-Answer Chain: Chains together context formatting, the QA prompt, and the LLM.
    question_answer_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | qa_prompt
        | llm
    )

    # 4. Conversational RAG Chain: The final chain that orchestrates the entire process.
    # It uses the history-aware retriever to get context and then the QA chain to answer.
    conversational_rag_chain = RunnableWithMessageHistory(
        RunnablePassthrough.assign(context=history_aware_retriever).assign(answer=question_answer_chain),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    st.success(f"Ready to answer questions about **{st.session_state.file_name}**.")
    
    if query := st.chat_input("Ask a question about the document:"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching for the answer..."):
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": query},
                        config={"configurable": {"session_id": st.session_state.file_key}}
                    )
                    answer = response["answer"].content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred during model invocation: {e}")