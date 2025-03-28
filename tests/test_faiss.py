#!/usr/bin/env python3
"""
Test script for FAISS vector database implementation.
This script tests the basic functionality of the FAISS vector store.
"""

import logging
import os
import sys
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("faiss_test")


def test_faiss_basic():
    """Test basic FAISS functionality: create, save, load, search"""
    logger.info("Starting basic FAISS test")

    # Initialize embedding function
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            task_type="retrieval_document"
        )
        logger.info("Embedding function initialized")
    except Exception as e:
        logger.error(f"Failed to initialize embedding function: {str(e)}")
        return False

    # Create test documents
    docs = [
        Document(page_content="FAISS is a library for efficient similarity search.",
                 metadata={"source": "test", "id": 1}),
        Document(page_content="Vector databases are used for semantic search applications.",
                 metadata={"source": "test", "id": 2}),
        Document(page_content="Embeddings convert text into numerical vectors.",
                 metadata={"source": "test", "id": 3}),
        Document(page_content="RAG combines retrieval with generative AI.",
                 metadata={"source": "test", "id": 4}),
    ]
    logger.info(f"Created {len(docs)} test documents")

    # Create vector store
    try:
        start_time = time.time()
        vectorstore = FAISS.from_documents(docs, embedding_function)
        logger.info(f"Vector store created in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        return False

    # Test saving
    test_dir = "test_faiss_db"
    os.makedirs(test_dir, exist_ok=True)
    try:
        start_time = time.time()
        vectorstore.save_local(test_dir)
        logger.info(f"Vector store saved in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to save vector store: {str(e)}")
        return False

    # Test loading
    try:
        start_time = time.time()
        loaded_vectorstore = FAISS.load_local(
            test_dir,
            embedding_function,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Vector store loaded in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load vector store: {str(e)}")
        return False

    # Test similarity search
    try:
        start_time = time.time()
        results = loaded_vectorstore.similarity_search(
            "What is semantic search?", k=2)
        search_time = time.time() - start_time
        logger.info(f"Similarity search completed in {search_time:.4f}s")
        logger.info(f"Found {len(results)} results")
        for i, doc in enumerate(results):
            logger.info(
                f"Result {i+1}: {doc.page_content[:50]}... (metadata: {doc.metadata})")
    except Exception as e:
        logger.error(f"Failed to perform similarity search: {str(e)}")
        return False

    # Test adding documents
    try:
        new_docs = [
            Document(page_content="LangChain is a framework for developing applications with LLMs.",
                     metadata={"source": "test", "id": 5}),
        ]
        start_time = time.time()
        loaded_vectorstore.add_documents(new_docs)
        logger.info(f"Added new documents in {time.time() - start_time:.2f}s")

        # Verify the document was added
        results = loaded_vectorstore.similarity_search(
            "What is LangChain?", k=1)
        logger.info(f"New document search result: {results[0].page_content}")
    except Exception as e:
        logger.error(f"Failed to add documents: {str(e)}")
        return False

    # Test MMR search
    try:
        start_time = time.time()
        results = loaded_vectorstore.max_marginal_relevance_search(
            "What are vector databases?",
            k=2,
            fetch_k=3,
            lambda_mult=0.5
        )
        search_time = time.time() - start_time
        logger.info(f"MMR search completed in {search_time:.4f}s")
        logger.info(f"Found {len(results)} diverse results")
        for i, doc in enumerate(results):
            logger.info(
                f"MMR Result {i+1}: {doc.page_content[:50]}... (metadata: {doc.metadata})")
    except Exception as e:
        logger.error(f"Failed to perform MMR search: {str(e)}")
        return False

    # Clean up
    try:
        import shutil
        shutil.rmtree(test_dir)
        logger.info(f"Cleaned up test directory: {test_dir}")
    except Exception as e:
        logger.error(f"Failed to clean up: {str(e)}")

    logger.info("Basic FAISS test completed successfully")
    return True


def test_faiss_deletion_simulation():
    """Test document deletion simulation with FAISS"""
    logger.info("Starting FAISS deletion simulation test")

    # Initialize embedding function
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_document"
    )

    # Create test documents with file_ids
    docs = []
    file_id_mapping = {}

    # File 1 documents
    file1_docs = [
        Document(page_content="Document 1 content part 1",
                 metadata={"source": "file1.txt", "file_id": 1, "page": 1}),
        Document(page_content="Document 1 content part 2",
                 metadata={"source": "file1.txt", "file_id": 1, "page": 2}),
    ]
    docs.extend(file1_docs)
    file_id_mapping[1] = file1_docs

    # File 2 documents
    file2_docs = [
        Document(page_content="Document 2 content part 1",
                 metadata={"source": "file2.txt", "file_id": 2, "page": 1}),
        Document(page_content="Document 2 content part 2",
                 metadata={"source": "file2.txt", "file_id": 2, "page": 2}),
    ]
    docs.extend(file2_docs)
    file_id_mapping[2] = file2_docs

    logger.info(
        f"Created {len(docs)} test documents across {len(file_id_mapping)} files")

    # Create vector store
    vectorstore = FAISS.from_documents(docs, embedding_function)
    logger.info("Vector store created")

    # Test search before deletion
    results = vectorstore.similarity_search("Document 1", k=2)
    logger.info("Search results before deletion:")
    for i, doc in enumerate(results):
        logger.info(
            f"Result {i+1}: {doc.page_content} (file_id: {doc.metadata.get('file_id')})")

    # Simulate deletion of file_id 1
    logger.info("Simulating deletion of file_id 1")

    # Get all documents except those with file_id 1
    remaining_docs = []
    for file_id, docs in file_id_mapping.items():
        if file_id != 1:
            remaining_docs.extend(docs)

    # Create new vector store with remaining documents
    new_vectorstore = FAISS.from_documents(remaining_docs, embedding_function)
    logger.info(
        f"Created new vector store with {len(remaining_docs)} documents")

    # Test search after deletion
    results = new_vectorstore.similarity_search("Document", k=2)
    logger.info("Search results after deletion:")
    for i, doc in enumerate(results):
        logger.info(
            f"Result {i+1}: {doc.page_content} (file_id: {doc.metadata.get('file_id')})")
        # Verify no documents from file_id 1 are returned
        assert doc.metadata.get(
            'file_id') != 1, "Deletion failed: found document with file_id 1"

    logger.info("FAISS deletion simulation test completed successfully")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("FAISS Vector Database Test")
    print("=" * 50)

    success = test_faiss_basic()
    if success:
        print("\n✅ Basic FAISS test passed")
    else:
        print("\n❌ Basic FAISS test failed")

    print("\n" + "=" * 50)

    success = test_faiss_deletion_simulation()
    if success:
        print("\n✅ FAISS deletion simulation test passed")
    else:
        print("\n❌ FAISS deletion simulation test failed")
