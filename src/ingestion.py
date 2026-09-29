import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def setup_pinecone_index():
    """Initializes the Pinecone index if it does not already exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536, # Matches text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

def run_ingestion(pdf_path: str):
    """Loads PDF, chunks text, embeds, and upserts into Pinecone."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}. Please download it first.")

    logger.info(f"Loading document from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    # PyPDFLoader automatically injects 'source' and 'page' into the Document metadata
    docs = loader.load()

    logger.info("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   # Meets the 500-1000 character requirement
        chunk_overlap=100 # Meets the 100 overlap requirement
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Generated {len(chunks)} chunks with metadata.")

    # Ensure index exists before upserting
    setup_pinecone_index()

    logger.info("Generating embeddings and upserting to Pinecone...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    logger.info("Ingestion complete!")
    return vector_store

if __name__ == "__main__":
    # When run as a script, execute ingestion on the default path
    run_ingestion("data/Ebook-Agentic-AI.pdf")
