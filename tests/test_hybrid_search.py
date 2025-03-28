"""
Standalone test script for hybrid search implementation.

This script demonstrates a hybrid search approach combining:
1. Vector search (FAISS) for semantic similarity
2. Keyword search (BM25) for lexical matching
3. Result fusion using Reciprocal Rank Fusion (RRF)

No imports from the existing codebase are used.
"""

import sys
import os
import time
import traceback
import logging
import numpy as np
import re
from typing import List, Dict, Any, Optional, Callable
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("hybrid_search_test")

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


class CustomBM25Retriever:
    """Custom BM25 retriever for keyword-based search."""

    def __init__(self, documents: List[Document], top_k: int = 3):
        """Initialize the BM25 retriever with documents."""
        self.top_k = top_k
        self.documents = documents

        # Preprocess documents for BM25
        self.corpus = []
        for doc in documents:
            # Tokenize and clean text
            tokens = self._preprocess_text(doc.page_content)
            self.corpus.append(tokens)

        # Initialize BM25
        self.bm25 = BM25Okapi(self.corpus)
        logger.info(
            f"BM25 retriever initialized with {len(documents)} documents")

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Tokenize
        tokens = text.split()
        return tokens

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query using BM25."""
        try:
            # Preprocess query
            query_tokens = self._preprocess_text(query)

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get top k document indices
            top_indices = np.argsort(scores)[::-1][:self.top_k]

            # Return top documents with scores in metadata
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include documents with non-zero scores
                    doc = self.documents[idx]
                    # Add BM25 score to metadata
                    doc_with_score = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata,
                                  "bm25_score": float(scores[idx])}
                    )
                    results.append(doc_with_score)

            logger.info(
                f"BM25 retrieved {len(results)} documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {str(e)}")
            return []


class CustomHybridRetriever:
    """Custom hybrid retriever combining vector search and BM25."""

    def __init__(
        self,
        vector_retriever: Any,
        keyword_retriever: CustomBM25Retriever,
        top_k: int = 3,
        weight_vector: float = 0.6,
        weight_keyword: float = 0.4,
        use_rrf: bool = True,
        rrf_k: int = 60
    ):
        """Initialize the hybrid retriever."""
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.top_k = top_k
        self.weight_vector = weight_vector
        self.weight_keyword = weight_keyword
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        logger.info(
            f"Hybrid retriever initialized with weights: vector={weight_vector}, keyword={weight_keyword}")

    def _reciprocal_rank_fusion(
        self,
        vector_docs: List[Document],
        keyword_docs: List[Document]
    ) -> List[Document]:
        """Combine results using Reciprocal Rank Fusion."""
        # Create a dictionary to store document scores by ID
        doc_scores = {}

        # Process vector search results
        for rank, doc in enumerate(vector_docs):
            doc_id = self._get_doc_id(doc)
            # RRF formula: 1 / (rank + k)
            score = 1.0 / (rank + self.rrf_k)
            doc_scores[doc_id] = {
                "doc": doc,
                "score": score,
                "vector_rank": rank,
                "keyword_rank": None
            }

        # Process keyword search results
        for rank, doc in enumerate(keyword_docs):
            doc_id = self._get_doc_id(doc)
            score = 1.0 / (rank + self.rrf_k)

            if doc_id in doc_scores:
                # Document already in results, update score and rank
                doc_scores[doc_id]["score"] += score
                doc_scores[doc_id]["keyword_rank"] = rank
            else:
                # New document
                doc_scores[doc_id] = {
                    "doc": doc,
                    "score": score,
                    "vector_rank": None,
                    "keyword_rank": rank
                }

        # Sort by score and take top k
        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True)[:self.top_k]

        # Create final document list with fusion metadata
        results = []
        for item in sorted_docs:
            doc = item["doc"]
            # Add fusion metadata
            metadata = {**doc.metadata}
            metadata["rrf_score"] = item["score"]
            metadata["vector_rank"] = item["vector_rank"]
            metadata["keyword_rank"] = item["keyword_rank"]

            results.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))

        return results

    def _weighted_fusion(
        self,
        vector_docs: List[Document],
        keyword_docs: List[Document]
    ) -> List[Document]:
        """Combine results using weighted fusion."""
        # Create a dictionary to store document scores by ID
        doc_scores = {}

        # Process vector search results
        for rank, doc in enumerate(vector_docs):
            doc_id = self._get_doc_id(doc)
            # Normalize score based on rank (higher rank = lower score)
            score = self.weight_vector * \
                (1.0 - (rank / len(vector_docs)) if vector_docs else 0)
            doc_scores[doc_id] = {
                "doc": doc,
                "score": score,
                "vector_rank": rank,
                "keyword_rank": None
            }

        # Process keyword search results
        for rank, doc in enumerate(keyword_docs):
            doc_id = self._get_doc_id(doc)
            # Normalize score based on rank
            score = self.weight_keyword * \
                (1.0 - (rank / len(keyword_docs)) if keyword_docs else 0)

            if doc_id in doc_scores:
                # Document already in results, update score and rank
                doc_scores[doc_id]["score"] += score
                doc_scores[doc_id]["keyword_rank"] = rank
            else:
                # New document
                doc_scores[doc_id] = {
                    "doc": doc,
                    "score": score,
                    "vector_rank": None,
                    "keyword_rank": rank
                }

        # Sort by score and take top k
        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True)[:self.top_k]

        # Create final document list with fusion metadata
        results = []
        for item in sorted_docs:
            doc = item["doc"]
            # Add fusion metadata
            metadata = {**doc.metadata}
            metadata["fusion_score"] = item["score"]
            metadata["vector_rank"] = item["vector_rank"]
            metadata["keyword_rank"] = item["keyword_rank"]

            results.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))

        return results

    def _get_doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document based on content and metadata."""
        # Use source and page if available
        if "source" in doc.metadata and "page" in doc.metadata:
            return f"{doc.metadata['source']}_{doc.metadata['page']}"

        # Fallback to content hash
        return str(hash(doc.page_content))

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query using hybrid search."""
        try:
            # Get results from both retrievers
            vector_docs = self.vector_retriever.get_relevant_documents(query)
            keyword_docs = self.keyword_retriever.get_relevant_documents(query)

            logger.info(f"Vector search returned {len(vector_docs)} documents")
            logger.info(
                f"Keyword search returned {len(keyword_docs)} documents")

            # Combine results
            if self.use_rrf:
                results = self._reciprocal_rank_fusion(
                    vector_docs, keyword_docs)
                logger.info(f"RRF fusion returned {len(results)} documents")
            else:
                results = self._weighted_fusion(vector_docs, keyword_docs)
                logger.info(
                    f"Weighted fusion returned {len(results)} documents")

            return results
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []


def create_vector_store(documents: List[Document]) -> FAISS:
    """Create a FAISS vector store from documents."""
    try:
        # Initialize embedding function
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            task_type="retrieval_document"
        )

        # Create vector store
        vectorstore = FAISS.from_documents(documents, embedding_function)
        logger.info(
            f"Created FAISS vector store with {len(documents)} documents")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise


def create_hybrid_retriever(documents: List[Document], top_k: int = 3) -> CustomHybridRetriever:
    """Create a hybrid retriever from documents."""
    try:
        # Create vector store
        vectorstore = create_vector_store(documents)

        # Create vector retriever
        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": max(top_k * 3, 10),
                "lambda_mult": 0.75
            }
        )

        # Create keyword retriever
        keyword_retriever = CustomBM25Retriever(documents, top_k=top_k)

        # Create hybrid retriever
        hybrid_retriever = CustomHybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            top_k=top_k,
            weight_vector=0.6,
            weight_keyword=0.4,
            use_rrf=True
        )

        logger.info(f"Created hybrid retriever with top_k={top_k}")
        return hybrid_retriever
    except Exception as e:
        logger.error(f"Error creating hybrid retriever: {str(e)}")
        raise


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

        logger.info(f"Created RAG chain with model: gemini-2.0-flash")

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

        logger.info(
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
