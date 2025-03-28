#!/usr/bin/env python3
"""
Comprehensive test script for the RAG workflow.
This script tests the entire workflow from document processing to question answering.
"""

import os
import sys
import time
import logging
import shutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import fitz  # PyMuPDF
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("rag_workflow_test")

# Test directories
TEST_DIR = "test_rag_workflow"
FAISS_DIR = os.path.join(TEST_DIR, "faiss_db")
IMAGE_DIR = os.path.join(TEST_DIR, "images")
SAMPLE_PDF = "doc-test.pdf"  # Use an existing PDF in the workspace

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def setup_test_environment():
    """Set up the test environment by creating necessary directories"""
    logger.info("Setting up test environment")
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    logger.info(f"Test directories created: {TEST_DIR}")


def cleanup_test_environment():
    """Clean up the test environment by removing test directories"""
    logger.info("Cleaning up test environment")
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    logger.info(f"Test directories removed: {TEST_DIR}")


def extract_images_from_pdf(pdf_path):
    """Extract images from a PDF file"""
    logger.info(f"Extracting images from {pdf_path}")
    images = []

    try:
        pdf_document = fitz.open(pdf_path)
        logger.info(f"PDF opened: {pdf_path} ({len(pdf_document)} pages)")

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Generate a unique filename
                image_filename = f"page{page_num+1}_img{img_index+1}.png"
                image_path = os.path.join(IMAGE_DIR, image_filename)

                # Save the image
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                # Add to our list
                images.append({
                    'path': image_path,
                    'page': page_num + 1,
                    'index': img_index + 1
                })

        logger.info(f"Extracted {len(images)} images from {pdf_path}")
        return images

    except Exception as e:
        logger.error(f"Error extracting images from {pdf_path}: {str(e)}")
        return []


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    logger.info(f"Extracting text from {pdf_path}")
    texts = []

    try:
        pdf_document = fitz.open(pdf_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()

            if text and text.strip():
                texts.append({
                    'content': text,
                    'page': page_num + 1
                })

        logger.info(f"Extracted text from {len(texts)} pages in {pdf_path}")
        return texts

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return []


def get_image_summaries(images):
    """Generate summaries for images using OpenAI GPT-4o"""
    logger.info(f"Generating summaries for {len(images)} images")
    summaries = []

    for img in images:
        try:
            # Read image
            with open(img['path'], 'rb') as img_file:
                image_bytes = img_file.read()

            # Convert to base64 for API
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Get summary from OpenAI GPT-4o
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a detailed image analyzer. Describe this image comprehensively, focusing on any text, diagrams, charts, or important visual elements."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this image in detail, focusing on any text, diagrams, charts, or important visual elements:"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }}
                    ]}
                ],
                max_tokens=1000
            )

            summary = response.choices[0].message.content
            logger.info(f"Generated summary for image on page {img['page']}")

            # Create document
            doc = Document(
                page_content=f"IMAGE: {summary}",
                metadata={
                    'page': img['page'],
                    'type': 'image',
                    'image_path': img['path'],
                    'file_id': 1  # Test file ID
                }
            )
            summaries.append(doc)

        except Exception as e:
            logger.error(
                f"Error generating summary for image {img['path']}: {str(e)}")

    logger.info(f"Generated {len(summaries)} image summaries")
    return summaries


def create_text_documents(texts):
    """Create Document objects from extracted text"""
    logger.info(f"Creating documents from {len(texts)} text chunks")
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_docs = [
        Document(
            page_content=text['content'],
            metadata={
                'page': text['page'],
                'file_id': 1,  # Test file ID
                'type': 'text',
                'source': SAMPLE_PDF
            }
        ) for text in texts
    ]

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
    )

    text_chunks = text_splitter.split_documents(text_docs)
    logger.info(f"Split text into {len(text_chunks)} chunks")
    return text_chunks


def create_faiss_index(documents):
    """Create a FAISS index from documents"""
    logger.info(f"Creating FAISS index with {len(documents)} documents")

    try:
        # Initialize embedding function
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            task_type="retrieval_document"
        )
        logger.info("Embedding function initialized")

        # Create vector store
        vectorstore = FAISS.from_documents(documents, embedding_function)
        logger.info("Vector store created")

        # Save the index
        vectorstore.save_local(FAISS_DIR)
        logger.info(f"Vector store saved to {FAISS_DIR}")

        return vectorstore

    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        return None


def test_mmr_search(vectorstore):
    """Test MMR search functionality"""
    logger.info("Testing MMR search")

    try:
        # Configure retriever with MMR
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 10,
                "lambda_mult": 0.75
            }
        )

        # Perform search
        query = "What is this document about?"
        results = retriever.invoke(query)

        logger.info(f"MMR search returned {len(results)} results")
        for i, doc in enumerate(results):
            logger.info(
                f"Result {i+1}: {doc.page_content[:50]}... (type: {doc.metadata.get('type')})")

        return True

    except Exception as e:
        logger.error(f"Error testing MMR search: {str(e)}")
        return False


def test_deletion(vectorstore):
    """Test document deletion functionality"""
    logger.info("Testing document deletion")

    try:
        # Get all documents
        all_docs = []
        for doc in vectorstore.docstore._dict.values():
            # Keep only docs that don't have file_id 1
            if doc.metadata.get('file_id') != 1:
                all_docs.append(doc)

        # If no documents left, add an initialization document
        if not all_docs:
            all_docs = [
                Document(page_content="Initialization document", metadata={"init": True})]

        # Create embedding function
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            task_type="retrieval_document"
        )

        # Create new vector store
        new_vectorstore = FAISS.from_documents(all_docs, embedding_function)

        # Save the new index
        new_vectorstore.save_local(FAISS_DIR)
        logger.info("Vector store updated after deletion")

        # Test search after deletion
        retriever = new_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 2})
        results = retriever.invoke("test query")

        logger.info(f"Search after deletion returned {len(results)} results")
        for i, doc in enumerate(results):
            file_id = doc.metadata.get('file_id')
            logger.info(
                f"Result {i+1}: {doc.page_content[:50]}... (file_id: {file_id})")
            # Verify no documents from file_id 1 are returned
            assert file_id != 1, "Deletion failed: found document with file_id 1"

        logger.info("Deletion test passed")
        return True

    except Exception as e:
        logger.error(f"Error testing deletion: {str(e)}")
        return False


def create_rag_chain(vectorstore):
    """Create a RAG chain for question answering"""
    logger.info("Creating RAG chain")

    try:
        # Configure retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.75
            }
        )

        # Initialize LLM using OpenAI
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2048
        )

        # Contextualization chain
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given chat history and a question, reformulate it to be standalone. 
            Consider both text and image contexts."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_prompt
        )

        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical expert analyzing documents. Use both text and image context.
            Text chunks may contain page numbers. Image summaries start with 'IMAGE:'. 
            Always cite sources using [page X] or [image X] notation."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("human", "Answer based on this context:\n{context}")
        ])

        # Assemble full chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)

        logger.info("RAG chain created successfully with OpenAI model")
        return retrieval_chain

    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
        return None


def test_qa(rag_chain):
    """Test question answering with the RAG chain"""
    logger.info("Testing question answering")

    try:
        # Test questions
        questions = [
            "What is this document about?",
            "Summarize the key points in the document",
            "What information is shown in the images?"
        ]

        # Empty chat history for testing
        chat_history = []

        for i, question in enumerate(questions):
            logger.info(f"Question {i+1}: {question}")

            # Invoke the chain
            result = rag_chain.invoke({
                "input": question,
                "chat_history": chat_history
            })

            logger.info(f"Answer: {result['answer'][:100]}...")

            # Update chat history
            chat_history.append({"role": "human", "content": question})
            chat_history.append(
                {"role": "assistant", "content": result["answer"]})

        logger.info("Question answering test completed")
        return True

    except Exception as e:
        logger.error(f"Error testing question answering: {str(e)}")
        return False


def run_workflow_test():
    """Run the complete workflow test"""
    try:
        # Setup
        setup_test_environment()

        # Extract content from PDF
        images = extract_images_from_pdf(SAMPLE_PDF)
        texts = extract_text_from_pdf(SAMPLE_PDF)

        # Process content
        image_summaries = get_image_summaries(images)
        text_chunks = create_text_documents(texts)

        # Combine documents
        all_docs = text_chunks + image_summaries
        logger.info(f"Total documents: {len(all_docs)}")

        # Create FAISS index
        vectorstore = create_faiss_index(all_docs)
        if not vectorstore:
            logger.error("Failed to create vector store")
            return False

        # Test MMR search
        if not test_mmr_search(vectorstore):
            logger.error("MMR search test failed")
            return False

        # Create RAG chain
        rag_chain = create_rag_chain(vectorstore)
        if not rag_chain:
            logger.error("Failed to create RAG chain")
            return False

        # Test question answering
        if not test_qa(rag_chain):
            logger.error("Question answering test failed")
            return False

        # Test deletion
        if not test_deletion(vectorstore):
            logger.error("Deletion test failed")
            return False

        logger.info("All tests passed successfully!")
        return True

    except Exception as e:
        logger.error(f"Workflow test failed: {str(e)}")
        return False
    finally:
        # Cleanup
        cleanup_test_environment()


if __name__ == "__main__":
    print("=" * 50)
    print("RAG Workflow Test")
    print("=" * 50)

    success = run_workflow_test()

    if success:
        print("\n✅ RAG workflow test passed")
    else:
        print("\n❌ RAG workflow test failed")
