from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

from . import config


def load_text_documents(text_dir: Path = config.TEXT_DIR) -> List[Document]:
    """Load all .txt files from a directory as LangChain Documents"""
    loader = DirectoryLoader(
        str(text_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    return loader.load()


def load_pdf_documents(pdf_dir: Path = config.PDF_DIR) -> List[Document]:
    """Load all .pdf files from a directory as LangChain Documents"""
    loader = DirectoryLoader(
        str(pdf_dir),
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False,
    )
    return loader.load()


def load_all_documents() -> List[Document]:
    """Load every text and PDF document under the data directory"""
    return load_text_documents() + load_pdf_documents()
