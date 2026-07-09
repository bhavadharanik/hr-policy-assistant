from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.rag.loader import load_policy_documents
from src.config import AI_EMBEDDING_MODEL

_vectorstore = None


def get_vectorstore():
    """Build or return cached FAISS vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model=AI_EMBEDDING_MODEL)
        chunks = load_policy_documents()
        _vectorstore = FAISS.from_documents(chunks, embeddings)
    return _vectorstore


def get_retriever(k: int = 4):
    """Return retriever over policy documents."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def similarity_search(query: str, k: int = 4):
    """Return full documents with metadata for source citation."""
    return get_vectorstore().similarity_search(query, k=k)
