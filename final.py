# app.py

import streamlit as st
import os
import logging
import tempfile
import requests
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import torch
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)

def is_gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()

# Constants
DOC_PATH = "res/computer_virus.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"
USE_GPU = is_gpu_available()


def ingest_pdf(doc_path):
    """Load PDF documents from a local path."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None

def download_pdf(url):
    """Download a file from the given URL to a temporary file."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
        tmp_file.write(response.content)
        tmp_file.close()
        logging.info("Downloaded file from URL.")
        return tmp_file.name
    except Exception as e:
        logging.error(f"Failed to download file: {e}")
        st.error("Failed to download file from the provided URL.")
        return None

def ingest_document_from_url(url):
    """Load a document from a URL (supports PDF, TXT, HTML, MD)."""
    tmp_path = download_pdf(url)
    if not tmp_path:
        return None
    ext = os.path.splitext(tmp_path)[1].lower()
    if ext == ".pdf":
        loader = UnstructuredPDFLoader(file_path=tmp_path)
    elif ext in [".txt", ".html", ".md"]:
        from langchain.document_loaders import UnstructuredFileLoader
        loader = UnstructuredFileLoader(file_path=tmp_path)
    else:
        logging.error("Unsupported file type.")
        st.error("Unsupported file type. Only PDF, TXT, HTML, and MD are supported.")
        os.remove(tmp_path)
        return None
    data = loader.load()
    os.remove(tmp_path)
    logging.info("Document loaded from URL successfully.")
    return data

def ingest_document_from_file(file_obj, file_name):
    """Load a document from an uploaded file (supports PDF, TXT, HTML, MD)."""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1])
    tmp_file.write(file_obj.read())
    tmp_file.close()
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        loader = UnstructuredPDFLoader(file_path=tmp_file.name)
    elif ext in [".txt", ".html", ".md"]:
        from langchain.document_loaders import UnstructuredFileLoader
        loader = UnstructuredFileLoader(file_path=tmp_file.name)
    else:
        logging.error("Unsupported file type.")
        st.error("Unsupported file type. Only PDF, TXT, HTML, and MD are supported.")
        os.remove(tmp_file.name)
        return None
    data = loader.load()
    os.remove(tmp_file.name)
    logging.info("Document loaded from uploaded file successfully.")
    return data


def split_documents(documents):
    """Split documents into smaller chunks."""
    # Optimize text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    # ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)  # Removed use_gpu parameter

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],

        template="""Generate 2 alternative phrasings for the question to improve vector search efficiency.
Original: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def create_llm(streamlit_container=None):
    """Create and return an Ollama LLM instance with proper error handling."""
    try:
        if streamlit_container:
            stream_handler = StreamHandler(streamlit_container)
            llm = ChatOllama(
                model=MODEL_NAME,
                temperature=0.1,
                streaming=True,
                callbacks=[stream_handler]
            )
        else:
            llm = ChatOllama(
                model=MODEL_NAME,
                temperature=0.1,
                streaming=False
            )
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        st.error("Failed to initialize the language model. Please check if Ollama is running.")
        return None

def main():
    st.title("Document Assistant")

    # Print if GPU is available
    if USE_GPU:
        st.write("GPU is available and will be used.")
    else:
        st.write("GPU is not available. Using CPU.")

    # Load the vector database
    vector_db = load_vector_db()
    if vector_db is None:
        st.error("Failed to load or create the vector database.")
        return

    # Section for file upload from the user interface
    st.markdown("### Upload New Knowledge Base Document")
    uploaded_file = st.file_uploader("Choose a file (PDF, TXT, HTML, MD)", type=["pdf", "txt", "html", "md"])
    if uploaded_file is not None:
        with st.spinner("Processing uploaded file..."):
            data = ingest_document_from_file(uploaded_file, uploaded_file.name)
            if data:
                new_chunks = split_documents(data)
                vector_db.add_documents(new_chunks)
                vector_db.persist()
                st.success("Uploaded document added to the knowledge base.")
            else:
                st.error("Failed to process the uploaded document.")

    # Section for adding documents via URL
    st.markdown("### Add New Knowledge Base Document from URL")
    url_input = st.text_input("Enter PDF/TXT/HTML/MD URL:")
    if st.button("Add Document") and url_input:
        with st.spinner("Adding document from URL..."):
            data = ingest_document_from_url(url_input)
            if data:
                new_chunks = split_documents(data)
                vector_db.add_documents(new_chunks)
                vector_db.persist()
                st.success("Document added to the knowledge base.")
            else:
                st.error("Failed to ingest the provided document.")

    # Section for Q&A
    user_input = st.text_input("Enter your question:", "")
    if user_input:
        response_container = st.empty()
        with st.spinner("Generating response..."):
            try:
                # Initialize the language model with streaming
                llm = create_llm(response_container)
                if llm is None:
                    return

                # Create the retriever and chain
                retriever = create_retriever(vector_db, llm)
                chain = create_chain(retriever, llm)

                # Get the response
                chain.invoke(user_input)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")


if __name__ == "__main__":
    main()
