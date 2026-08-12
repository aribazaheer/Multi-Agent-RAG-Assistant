from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Paths
DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"


def load_documents():
    documents = []

    for file_path in DATA_DIR.iterdir():

        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))

        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )

        elif file_path.suffix.lower() == ".md":
            loader = UnstructuredMarkdownLoader(str(file_path))

        else:
            continue

        documents.extend(loader.load())

    return documents


def main():
    print("Loading documents...")

    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    # Local Ollama embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Store vectors in ChromaDB
    vectorstore = Chroma(
        collection_name="day15_documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vectorstore.add_documents(chunks)

    print("Documents successfully stored in ChromaDB!")


if __name__ == "__main__":
    main()