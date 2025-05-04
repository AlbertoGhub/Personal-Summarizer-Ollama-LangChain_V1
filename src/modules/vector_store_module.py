from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=7500, chunk_overlap=100):
    """
    Function to chunk documents using RecursiveCharacterTextSplitter.
    """
    print('Chunking the document...')
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print('Document chunked successfully!!!')
    return chunks

def add_to_vector_store(chunks, model_name='mxbai-embed-large', collection_name='rag-collection'):
    """
    Function to add chunks of documents to the Chroma vector store.
    """
    print('Adding to vector database...')
    embedding_function = OllamaEmbeddings(model=model_name)

    # Initialize the Chroma vector store
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function
    )

    # Add documents to the vector store
    vector_store.add_documents(chunks)
    print('Document added to vector database successfully!!!')

    return vector_store