
import re
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# The embedding model is shared (it holds no user data), so caching it globally is safe.
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


def _clean(text: str) -> str:
    """Collapse newlines/extra spaces that PDFs leave behind."""
    return re.sub(r"\s+", " ", text).strip()


def ingest_resume(pdf_path: str):
    """
    Read a resume PDF and build a fresh, in-memory vector store for it.

    Returns: (vectorstore, resume_full_text, num_chunks)

    Nothing is written to disk and nothing is global, so an old resume can
    never leak into a new analysis (and users don't share data).
    """
    pages = PyPDFLoader(pdf_path).load()

    resume_text = _clean(" ".join(p.page_content for p in pages))
    if len(resume_text) < 50:
        raise ValueError(
            "No readable text found in this PDF. It may be a scanned image; "
            "please upload a text-based PDF."
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(pages)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"resume_{uuid.uuid4().hex}",  # unique per upload
    )
    return vectorstore, resume_text, len(chunks)