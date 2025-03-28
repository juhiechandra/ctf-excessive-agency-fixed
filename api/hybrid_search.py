"""
Hybrid Search Implementation for RAG System

This module implements a hybrid search approach combining:
1. Vector search (FAISS) for semantic similarity
2. BM25 for keyword-based relevance
3. Result fusion using Reciprocal Rank Fusion (RRF)

The hybrid approach provides better retrieval performance by leveraging
both semantic understanding and keyword matching.
"""

import os
import time
import numpy as np
import re
from typing import List, Dict, Any, Optional, Callable
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from logger import model_logger, error_logger, PerformanceTimer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CustomBM25Retriever:
    """Custom BM25 retriever for keyword-based search."""

    def __init__(self, documents: List[Document], top_k: int = 5):
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
        model_logger.info(
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
            with PerformanceTimer(model_logger, "BM25 retrieval"):
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

                model_logger.info(
                    f"BM25 retrieved {len(results)} documents for query: {query[:50]}...")
                return results
        except Exception as e:
            error_msg = f"Error in BM25 retrieval: {str(e)}"
            model_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return []


class CustomHybridRetriever:
    """Custom hybrid retriever combining vector search and BM25."""

    def __init__(
        self,
        vector_retriever: Any,
        keyword_retriever: CustomBM25Retriever,
        top_k: int = 5,
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
        model_logger.info(
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
        # Use file_id and page if available for compatibility with existing system
        if "file_id" in doc.metadata and "page" in doc.metadata:
            return f"{doc.metadata['file_id']}_{doc.metadata['page']}"

        # Use source and page if available (from test implementation)
        if "source" in doc.metadata and "page" in doc.metadata:
            return f"{doc.metadata['source']}_{doc.metadata['page']}"

        # Fallback to content hash
        return str(hash(doc.page_content))

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query using hybrid search."""
        try:
            with PerformanceTimer(model_logger, "Hybrid retrieval"):
                # Get results from both retrievers
                vector_docs = self.vector_retriever.get_relevant_documents(
                    query)
                keyword_docs = self.keyword_retriever.get_relevant_documents(
                    query)

                model_logger.info(
                    f"Vector search returned {len(vector_docs)} documents")
                model_logger.info(
                    f"Keyword search returned {len(keyword_docs)} documents")

                # Combine results
                if self.use_rrf:
                    results = self._reciprocal_rank_fusion(
                        vector_docs, keyword_docs)
                    model_logger.info(
                        f"RRF fusion returned {len(results)} documents")
                else:
                    results = self._weighted_fusion(vector_docs, keyword_docs)
                    model_logger.info(
                        f"Weighted fusion returned {len(results)} documents")

                return results
        except Exception as e:
            error_msg = f"Error in hybrid retrieval: {str(e)}"
            model_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return []


class BM25Retriever(BaseRetriever):
    """LangChain compatible BM25 retriever."""

    k: int = 5  # Define k as a class attribute for Pydantic

    def __init__(self, documents: List[Document], k: int = 5):
        """Initialize the BM25 retriever with documents."""
        super().__init__()
        self._custom_retriever = CustomBM25Retriever(documents, top_k=k)
        self.k = k  # This will now work since k is defined as a class attribute
        model_logger.info(f"LangChain BM25Retriever initialized with k={k}")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to the query using BM25."""
        return self._custom_retriever.get_relevant_documents(query)


class HybridRetriever(BaseRetriever):
    """LangChain compatible hybrid retriever."""

    k: int = 5  # Define k as a class attribute for Pydantic
    weight_vector: float = 0.6
    weight_keyword: float = 0.4
    use_rrf: bool = True
    rrf_k: int = 60

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BaseRetriever,
        k: int = 5,
        weight_vector: float = 0.6,
        weight_keyword: float = 0.4,
        use_rrf: bool = True,
        rrf_k: int = 60
    ):
        """Initialize the hybrid retriever."""
        super().__init__()

        # If keyword_retriever is a BM25Retriever, use its internal retriever
        if isinstance(keyword_retriever, BM25Retriever):
            keyword_retriever_internal = keyword_retriever._custom_retriever
        elif isinstance(keyword_retriever, CustomBM25Retriever):
            keyword_retriever_internal = keyword_retriever
        else:
            model_logger.warning(
                f"Unexpected keyword retriever type: {type(keyword_retriever)}")
            keyword_retriever_internal = keyword_retriever

        self._custom_retriever = CustomHybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever_internal,
            top_k=k,
            weight_vector=weight_vector,
            weight_keyword=weight_keyword,
            use_rrf=use_rrf,
            rrf_k=rrf_k
        )
        # Set attributes using the defined class attributes
        self.k = k
        self.weight_vector = weight_vector
        self.weight_keyword = weight_keyword
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        model_logger.info(
            f"LangChain HybridRetriever initialized with k={k}, weights: vector={weight_vector}, keyword={weight_keyword}")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to the query using hybrid search."""
        return self._custom_retriever.get_relevant_documents(query)


def create_vector_store(documents: List[Document]) -> FAISS:
    """Create a FAISS vector store from documents."""
    try:
        with PerformanceTimer(model_logger, "create_vector_store"):
            # Initialize embedding function
            embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                task_type="retrieval_document"
            )

            # Create vector store
            vectorstore = FAISS.from_documents(documents, embedding_function)
            model_logger.info(
                f"Created FAISS vector store with {len(documents)} documents")
            return vectorstore
    except Exception as e:
        error_msg = f"Error creating vector store: {str(e)}"
        model_logger.error(error_msg)
        error_logger.error(error_msg, exc_info=True)
        raise


def create_hybrid_retriever(documents: List[Document], top_k: int = 5) -> CustomHybridRetriever:
    """Create a hybrid retriever from documents."""
    try:
        with PerformanceTimer(model_logger, "create_hybrid_retriever"):
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

            model_logger.info(f"Created hybrid retriever with top_k={top_k}")
            return hybrid_retriever
    except Exception as e:
        error_msg = f"Error creating hybrid retriever: {str(e)}"
        model_logger.error(error_msg)
        error_logger.error(error_msg, exc_info=True)
        raise


def create_hybrid_retriever_from_faiss(
    vectorstore: FAISS,
    documents: List[Document] = None,
    k: int = 5,
    weight_vector: float = 0.6,
    weight_keyword: float = 0.4,
    use_rrf: bool = True,
    rrf_k: int = 60
) -> HybridRetriever:
    """
    Create a hybrid retriever from a FAISS vectorstore.

    Args:
        vectorstore: The FAISS vectorstore to use for vector search
        documents: The documents to use for BM25 search (if None, will use vectorstore.docstore.docs)
        k: The number of documents to retrieve
        weight_vector: The weight to give to vector search results
        weight_keyword: The weight to give to keyword search results
        use_rrf: Whether to use Reciprocal Rank Fusion (RRF) for combining results
        rrf_k: The k parameter for RRF

    Returns:
        A HybridRetriever instance
    """
    try:
        with PerformanceTimer(model_logger, "create_hybrid_retriever_from_faiss"):
            # Create vector retriever
            vector_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": max(k * 3, 10),
                    "lambda_mult": 0.75
                }
            )

            # Get documents from vectorstore if not provided
            if documents is None:
                try:
                    # Try to get documents from vectorstore using different methods
                    try:
                        # First try the docstore.docs approach
                        documents = list(vectorstore.docstore.docs.values())
                    except AttributeError:
                        # If that fails, try the docstore._dict approach
                        try:
                            documents = []
                            for doc_id, doc in vectorstore.docstore._dict.items():
                                # Handle both Document objects and tuples
                                if isinstance(doc, Document):
                                    documents.append(doc)
                                elif isinstance(doc, tuple) and len(doc) > 0:
                                    # If it's a tuple, the first element is usually the Document
                                    if isinstance(doc[0], Document):
                                        documents.append(doc[0])
                        except AttributeError:
                            # If that also fails, try the get_all_documents method
                            try:
                                documents = vectorstore.get_all_documents()
                            except (AttributeError, NotImplementedError):
                                # If all methods fail, raise an error
                                raise ValueError(
                                    "Could not extract documents from vectorstore")

                    model_logger.info(
                        f"Using {len(documents)} documents from vectorstore")
                except Exception as e:
                    error_msg = f"Error getting documents from vectorstore: {str(e)}"
                    model_logger.error(error_msg)
                    error_logger.error(error_msg, exc_info=True)
                    # Return just the vector retriever wrapped in a class that matches the HybridRetriever interface
                    model_logger.warning(
                        "Falling back to vector-only retrieval due to missing documents")

                    class WrappedVectorRetriever(BaseRetriever):
                        def _get_relevant_documents(
                            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
                        ) -> List[Document]:
                            return vector_retriever.get_relevant_documents(query)

                    return WrappedVectorRetriever()

            # Create BM25 retriever
            keyword_retriever = BM25Retriever(documents, k=k)

            # Create hybrid retriever
            hybrid_retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                keyword_retriever=keyword_retriever,
                k=k,
                weight_vector=weight_vector,
                weight_keyword=weight_keyword,
                use_rrf=use_rrf,
                rrf_k=rrf_k
            )

            model_logger.info(f"Created hybrid retriever with k={k}")
            return hybrid_retriever

    except Exception as e:
        error_msg = f"Error creating hybrid retriever: {str(e)}"
        model_logger.error(error_msg)
        error_logger.error(error_msg, exc_info=True)
        raise
