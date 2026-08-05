"""Build the vector store from the PDFs and text files under data/."""

from pipeline.embeddings import EmbeddingManager
from pipeline.loaders import load_all_documents
from pipeline.splitter import split_documents
from pipeline.vector_store import VectorStore


def main():
    documents = load_all_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embedding_manager = EmbeddingManager()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)
    print(f"Vector store now has {vector_store.count()} documents")


if __name__ == "__main__":
    main()
