from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from faiss_utils import vectorstore
from hybrid_search import create_hybrid_retriever_from_faiss, create_hybrid_retriever
from dotenv import load_dotenv
from logger import model_logger, error_logger, PerformanceTimer
import os
import time

load_dotenv()

model_logger.info("Initializing LangChain utilities")


def get_rag_chain(model="gemini-2.0-flash", use_hybrid_search=True):
    """
    Create a RAG chain with the specified model.

    Args:
        model (str): The model to use for the RAG chain.
        use_hybrid_search (bool): Whether to use hybrid search (vector + BM25) or just vector search.

    Returns:
        A LangChain retrieval chain.
    """
    with PerformanceTimer(model_logger, f"get_rag_chain:{model}"):
        try:
            # Configure retriever
            model_logger.info(f"Configuring retriever for model: {model}")

            if use_hybrid_search:
                # Use hybrid search (vector + BM25)
                model_logger.info("Using hybrid search (vector + BM25)")
                retriever = create_hybrid_retriever_from_faiss(
                    vectorstore=vectorstore,
                    k=6,
                    weight_vector=0.6,
                    weight_keyword=0.4,
                    use_rrf=True,
                    rrf_k=60
                )
                model_logger.info("Hybrid retriever configured")
            else:
                # Use vector search only
                model_logger.info("Using vector search only")
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 6,
                        "fetch_k": 20,
                        "lambda_mult": 0.75
                    }
                )
                model_logger.info(
                    "Vector retriever configured with MMR search")

            # Initialize LLM - ONLY USE GEMINI MODELS
            # Force model to be a Gemini model
            if not model.startswith("gemini"):
                model_logger.warning(
                    f"Non-Gemini model requested: {model}, forcing to gemini-2.0-flash")
                model = "gemini-2.0-flash"

            model_logger.info(f"Using Gemini model: {model}")
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.7,
                top_k=40,
                max_output_tokens=2048
            )

            # Contextualization prompt
            model_logger.info("Creating contextualization prompt")
            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", """Given chat history and a question, reformulate it to be standalone."""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            model_logger.info("Creating history-aware retriever")
            history_aware_retriever = create_history_aware_retriever(
                llm,
                retriever,
                contextualize_prompt
            )

            # QA prompt - simplified for basic RAG
            model_logger.info("Creating QA prompt")
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions based on the provided context from documents.
                Answer the user's questions based on the information provided in the context. 
                If the information is not in the context, say so politely."""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                ("human", "Context from documents:\n{context}")
            ])

            # Assemble full chain
            model_logger.info("Creating question-answer chain")
            question_answer_chain = create_stuff_documents_chain(
                llm, qa_prompt)

            model_logger.info("Creating retrieval chain")
            retrieval_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )

            model_logger.info(
                f"RAG chain created successfully for model: {model}")
            return retrieval_chain

        except Exception as e:
            error_msg = f"Error creating RAG chain for model {model}: {str(e)}"
            model_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise
