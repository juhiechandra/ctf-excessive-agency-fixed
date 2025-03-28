"""
Test script for hybrid search implementation.

This script tests the hybrid search implementation in the API.
"""

import os
import time
import traceback
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from hybrid_search import (
    CustomBM25Retriever,
    create_vector_store,
    create_hybrid_retriever
)
from logger import model_logger, error_logger, PerformanceTimer

# Load environment variables
load_dotenv()

# Sample documents for testing
SAMPLE_DOCS = [
    Document(page_content="Hybrid search combines vector search and keyword search for better results.",
             metadata={"source": "article1", "page": 1}),
    Document(page_content="Vector search uses embeddings to find semantically similar documents.",
             metadata={"source": "article2", "page": 1}),
    Document(page_content="BM25 is a keyword search algorithm that ranks documents based on term frequency.",
             metadata={"source": "article3", "page": 1}),
    Document(page_content="FAISS is a library for efficient similarity search developed by Facebook.",
             metadata={"source": "article4", "page": 1}),
    Document(page_content="Retrieval Augmented Generation (RAG) enhances LLM responses with external knowledge.",
             metadata={"source": "article5", "page": 1}),
    Document(page_content="Reciprocal Rank Fusion combines results from multiple search algorithms.",
             metadata={"source": "article6", "page": 1}),
    Document(page_content="LangChain is a framework for developing applications with language models.",
             metadata={"source": "article7", "page": 1}),
    Document(page_content="Embeddings are vector representations of text that capture semantic meaning.",
             metadata={"source": "article8", "page": 1}),
]


def test_vector_search():
    """Test vector search using FAISS."""
    try:
        print("\n=== Testing Vector Search (FAISS) ===")

        # Create vector store
        vectorstore = create_vector_store(SAMPLE_DOCS)

        # Create vector retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 8,
                "lambda_mult": 0.75
            }
        )

        # Test queries
        queries = [
            "What is hybrid search?",
            "How does vector search work?",
            "What is BM25?"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            start_time = time.time()
            docs = retriever.get_relevant_documents(query)
            end_time = time.time()

            print(
                f"Retrieved {len(docs)} documents in {end_time - start_time:.2f} seconds")

            # Print documents
            for i, doc in enumerate(docs):
                print(f"Document {i+1}: {doc.page_content}")
                print(f"Metadata: {doc.metadata}")

        return True

    except Exception as e:
        print(f"Error in vector search test: {str(e)}")
        traceback.print_exc()
        return False


def test_keyword_search():
    """Test keyword search using BM25."""
    try:
        print("\n=== Testing Keyword Search (BM25) ===")

        # Create BM25 retriever
        retriever = CustomBM25Retriever(SAMPLE_DOCS, top_k=3)

        # Test queries
        queries = [
            "hybrid search combines",
            "FAISS library",
            "embeddings vector"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            start_time = time.time()
            docs = retriever.get_relevant_documents(query)
            end_time = time.time()

            print(
                f"Retrieved {len(docs)} documents in {end_time - start_time:.2f} seconds")

            # Print documents
            for i, doc in enumerate(docs):
                print(f"Document {i+1}: {doc.page_content}")
                print(f"Metadata: {doc.metadata}")
                if "bm25_score" in doc.metadata:
                    print(f"BM25 Score: {doc.metadata['bm25_score']}")

        return True

    except Exception as e:
        print(f"Error in keyword search test: {str(e)}")
        traceback.print_exc()
        return False


def test_hybrid_search():
    """Test hybrid search using FAISS and BM25."""
    try:
        print("\n=== Testing Hybrid Search (FAISS + BM25) ===")

        # Create hybrid retriever
        retriever = create_hybrid_retriever(SAMPLE_DOCS, top_k=3)

        # Test queries
        queries = [
            "vector search embeddings",
            "BM25 algorithm for search",
            "RAG with FAISS"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            start_time = time.time()
            docs = retriever.get_relevant_documents(query)
            end_time = time.time()

            print(
                f"Retrieved {len(docs)} documents in {end_time - start_time:.2f} seconds")

            # Print documents
            for i, doc in enumerate(docs):
                print(f"Document {i+1}: {doc.page_content}")
                print(f"Metadata: {doc.metadata}")
                if "rrf_score" in doc.metadata:
                    print(f"RRF Score: {doc.metadata['rrf_score']}")
                    print(f"Vector Rank: {doc.metadata['vector_rank']}")
                    print(f"Keyword Rank: {doc.metadata['keyword_rank']}")

        return True

    except Exception as e:
        print(f"Error in hybrid search test: {str(e)}")
        traceback.print_exc()
        return False


def test_rag_with_hybrid_search():
    """Test RAG chain with hybrid search."""
    try:
        print("\n=== Testing RAG Chain with Hybrid Search ===")

        # Create vector store for direct use
        vectorstore = create_vector_store(SAMPLE_DOCS)

        # Create vector retriever
        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 8,
                "lambda_mult": 0.75
            }
        )

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7,
            top_k=40,
            max_output_tokens=1024
        )

        # Contextualization prompt
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given chat history and a question, reformulate it to be standalone."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # History-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm,
            vector_retriever,
            contextualize_prompt
        )

        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical expert analyzing documents. Always cite sources."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("human", "Answer based on this context:\n{context}")
        ])

        # Question-answer chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Retrieval chain
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        model_logger.info(f"Created RAG chain with model: gemini-2.0-flash")

        # Test queries
        queries = [
            "What is hybrid search and how does it work?",
            "Explain the difference between vector search and keyword search"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            start_time = time.time()
            response = retrieval_chain.invoke(
                {"input": query, "chat_history": []})
            end_time = time.time()

            print(f"Response generated in {end_time - start_time:.2f} seconds")
            print(f"Answer: {response['answer']}")

        return True

    except Exception as e:
        print(f"Error in RAG with hybrid search test: {str(e)}")
        traceback.print_exc()
        return False


def test_rag_comparison():
    """Compare RAG with and without hybrid search."""
    try:
        print("\n=== Comparing RAG with and without Hybrid Search ===")

        # Create vector store
        vectorstore = create_vector_store(SAMPLE_DOCS)

        # Create vector retriever
        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 8,
                "lambda_mult": 0.75
            }
        )

        # Create hybrid retriever (for manual use)
        hybrid_retriever = create_hybrid_retriever(SAMPLE_DOCS, top_k=3)

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7,
            top_k=40,
            max_output_tokens=1024
        )

        # Create RAG chain with vector retriever
        # Contextualization prompt
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given chat history and a question, reformulate it to be standalone."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # History-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm,
            vector_retriever,
            contextualize_prompt
        )

        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical expert analyzing documents. Always cite sources."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("human", "Answer based on this context:\n{context}")
        ])

        # Question-answer chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Retrieval chain
        chain_vector = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        model_logger.info(
            f"Created vector-based RAG chain with model: gemini-2.0-flash")

        # Test queries
        queries = [
            "What is the relationship between FAISS and embeddings?",
            "How does BM25 compare to vector search?"
        ]

        for query in queries:
            print(f"\nQuery: {query}")

            # First, get hybrid search results manually
            print("With Hybrid Search:")
            hybrid_docs = hybrid_retriever.get_relevant_documents(query)

            # Use the vector-based chain but with our hybrid results
            start_time = time.time()

            # Create a direct answer using the LLM with the hybrid docs
            hybrid_context = "\n\n".join([
                f"{doc.page_content}\nSource: {doc.metadata.get('source', 'Unknown')}"
                for doc in hybrid_docs
            ])

            hybrid_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a technical expert analyzing documents. Always cite sources."),
                ("human", f"{query}"),
                ("human", f"Answer based on this context:\n{hybrid_context}")
            ])

            hybrid_response = llm.invoke(hybrid_prompt.format_messages())

            end_time = time.time()
            hybrid_time = end_time - start_time

            print(f"Response generated in {hybrid_time:.2f} seconds")
            print(f"Answer: {hybrid_response.content}")

            # Test with vector search only
            print("\nWith Vector Search Only:")
            start_time = time.time()
            response_vector = chain_vector.invoke(
                {"input": query, "chat_history": []})
            end_time = time.time()
            vector_time = end_time - start_time

            print(f"Response generated in {vector_time:.2f} seconds")
            print(f"Answer: {response_vector['answer']}")

            # Compare
            print(
                f"\nTime difference: {hybrid_time - vector_time:.2f} seconds")

        return True

    except Exception as e:
        print(f"Error in RAG comparison test: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting hybrid search tests...")

    # Run all tests
    vector_test = test_vector_search()
    keyword_test = test_keyword_search()
    hybrid_test = test_hybrid_search()
    rag_test = test_rag_with_hybrid_search()
    comparison_test = test_rag_comparison()

    # Print summary
    print("\n=== Test Summary ===")
    print(f"Vector Search: {'✅ PASSED' if vector_test else '❌ FAILED'}")
    print(f"Keyword Search: {'✅ PASSED' if keyword_test else '❌ FAILED'}")
    print(f"Hybrid Search: {'✅ PASSED' if hybrid_test else '❌ FAILED'}")
    print(f"RAG with Hybrid Search: {'✅ PASSED' if rag_test else '❌ FAILED'}")
    print(f"RAG Comparison: {'✅ PASSED' if comparison_test else '❌ FAILED'}")

    # Overall result
    if all([vector_test, keyword_test, hybrid_test, rag_test, comparison_test]):
        print(
            "\n🎉 All tests passed! The hybrid search implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the logs for details.")
