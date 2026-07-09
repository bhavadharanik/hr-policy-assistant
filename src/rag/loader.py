import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import KNOWLEDGE_BASE_DIR


def load_policy_documents():
    """Load all markdown policy documents from the knowledge base directory."""
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} policy documents, split into {len(chunks)} chunks")
    return chunks
