from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict, Tuple
from langchain_core.documents import Document
import fitz  # PyMuPDF
import pdfplumber
import google.generativeai as genai
from openai import OpenAI
import os
import base64
from datetime import datetime
import traceback
import shutil
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from logger import model_logger, error_logger, PerformanceTimer

# Load environment variables
load_dotenv()

# Configure APIs
model_logger.info("Configuring API clients")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FAISS
faiss_db_path = "./faiss_db"
os.makedirs(faiss_db_path, exist_ok=True)
model_logger.info(f"FAISS DB path: {faiss_db_path}")

# Document collection path
collection_path = os.path.join(faiss_db_path, "document_collection")
os.makedirs(collection_path, exist_ok=True)

# File ID to document mapping for deletion tracking
file_id_mapping = {}

try:
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_document"
    )
    model_logger.info("Embedding function initialized")
except Exception as e:
    error_logger.error(
        f"Failed to initialize embedding function: {str(e)}", exc_info=True)
    raise

# Initialize or load the vector store
try:
    if os.path.exists(os.path.join(collection_path, "index.faiss")):
        model_logger.info("Loading existing FAISS index")
        vectorstore = FAISS.load_local(
            collection_path,
            embedding_function,
            allow_dangerous_deserialization=True
        )
        model_logger.info("FAISS vector store loaded")
    else:
        model_logger.info("Creating new FAISS vector store")
        # Initialize with empty documents list
        vectorstore = FAISS.from_documents(
            [Document(page_content="Initialization document",
                      metadata={"init": True})],
            embedding_function
        )
        # Save the initial index
        vectorstore.save_local(collection_path)
        model_logger.info("New FAISS vector store initialized")
except Exception as e:
    error_logger.error(
        f"Failed to initialize vector store: {str(e)}", exc_info=True)
    raise


def extract_images_pymupdf(pdf_path: str, output_dir: str) -> List[Dict]:
    """Extract images from PDF using PyMuPDF"""
    with PerformanceTimer(model_logger, f"extract_images_pymupdf:{os.path.basename(pdf_path)}"):
        images = []
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                error_msg = f"PDF file does not exist: {pdf_path}"
                model_logger.error(error_msg)
                error_logger.error(error_msg)
                return []

            # Check file size
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                error_msg = f"PDF file is empty (0 bytes): {pdf_path}"
                model_logger.error(error_msg)
                error_logger.error(error_msg)
                return []

            model_logger.info(
                f"Opening PDF file for image extraction: {pdf_path} ({file_size} bytes)")

            # Check file extension
            _, file_extension = os.path.splitext(pdf_path)
            if file_extension.lower() != '.pdf':
                error_msg = f"File is not a PDF: {pdf_path} (extension: {file_extension})"
                model_logger.error(error_msg)
                error_logger.error(error_msg)
                return []

            pdf_document = fitz.open(pdf_path)
            model_logger.info(
                f"PDF opened: {pdf_path} ({len(pdf_document)} pages)")

            if len(pdf_document) == 0:
                error_msg = f"PDF has no pages: {pdf_path}"
                model_logger.error(error_msg)
                error_logger.error(error_msg)
                return []

            for page_num in range(len(pdf_document)):
                try:
                    page = pdf_document[page_num]
                    image_list = page.get_images(full=True)

                    if not image_list:
                        model_logger.info(
                            f"No images found on page {page_num+1}")
                        continue

                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]

                            # Generate a unique filename
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            image_filename = f"page{page_num+1}_img{img_index+1}_{timestamp}.png"
                            image_path = os.path.join(
                                output_dir, image_filename)

                            # Save the image
                            with open(image_path, "wb") as img_file:
                                img_file.write(image_bytes)

                            # Add to our list
                            images.append({
                                'path': image_path,
                                'page': page_num + 1,
                                'index': img_index + 1
                            })
                        except Exception as img_error:
                            error_msg = f"Error extracting image {img_index+1} from page {page_num+1}: {str(img_error)}"
                            model_logger.error(error_msg)
                            error_logger.error(error_msg, exc_info=True)
                            # Continue with other images
                except Exception as page_error:
                    error_msg = f"Error processing page {page_num+1} for images: {str(page_error)}"
                    model_logger.error(error_msg)
                    error_logger.error(error_msg, exc_info=True)
                    # Continue with other pages

            model_logger.info(
                f"Extracted {len(images)} images from {pdf_path}")
            return images

        except Exception as e:
            error_msg = f"Error extracting images from {pdf_path}: {str(e)}"
            model_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return []


def extract_text_pdfplumber(pdf_path: str) -> List[Dict]:
    """Extract text from PDF using pdfplumber"""
    with PerformanceTimer(model_logger, f"extract_text_pdfplumber:{os.path.basename(pdf_path)}"):
        texts = []
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                error_msg = f"PDF file does not exist: {pdf_path}"
                model_logger.error(error_msg)
                error_logger.error(error_msg)
                return []

            # Check file size
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                error_msg = f"PDF file is empty (0 bytes): {pdf_path}"
                model_logger.error(error_msg)
                error_logger.error(error_msg)
                return []

            model_logger.info(
                f"Opening PDF file: {pdf_path} ({file_size} bytes)")

            # Check file extension
            _, file_extension = os.path.splitext(pdf_path)
            if file_extension.lower() != '.pdf':
                error_msg = f"File is not a PDF: {pdf_path} (extension: {file_extension})"
                model_logger.error(error_msg)
                error_logger.error(error_msg)
                return []

            with pdfplumber.open(pdf_path) as pdf:
                model_logger.info(
                    f"PDF opened with pdfplumber: {pdf_path} ({len(pdf.pages)} pages)")

                if len(pdf.pages) == 0:
                    error_msg = f"PDF has no pages: {pdf_path}"
                    model_logger.error(error_msg)
                    error_logger.error(error_msg)
                    return []

                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            texts.append({
                                'content': text,
                                'page': page_num + 1,
                                'text': text  # Add text field for compatibility
                            })
                        else:
                            model_logger.warning(
                                f"Page {page_num+1} has no text content")
                    except Exception as page_error:
                        error_msg = f"Error extracting text from page {page_num+1}: {str(page_error)}"
                        model_logger.error(error_msg)
                        error_logger.error(error_msg, exc_info=True)
                        # Continue with other pages

            if not texts:
                error_msg = f"No text could be extracted from any page in {pdf_path}"
                model_logger.error(error_msg)
                error_logger.error(error_msg)
                return []

            model_logger.info(
                f"Extracted text from {len(texts)} pages in {pdf_path}")
            return texts

        except Exception as e:
            error_msg = f"Error extracting text from {pdf_path}: {str(e)}"
            model_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return []


def resize_image(image_path: str, max_width: int = 800, max_height: int = 800, quality: int = 85) -> bytes:
    """Resize an image to reduce its size while maintaining readability"""
    try:
        with Image.open(image_path) as img:
            # Calculate new dimensions while maintaining aspect ratio
            width, height = img.size
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)

            # Convert to RGB if image is in RGBA mode (handles PNG transparency)
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # Save to bytes
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            return buffer.getvalue()
    except Exception as e:
        error_msg = f"Error resizing image {image_path}: {str(e)}"
        error_logger.error(error_msg, exc_info=True)
        # Return original image if resize fails
        with open(image_path, 'rb') as img_file:
            return img_file.read()


def get_image_summaries(images: List[Dict]) -> List[Document]:
    """Generate summaries for images using OpenAI GPT-4o"""
    with PerformanceTimer(model_logger, f"get_image_summaries:{len(images)} images"):
        summaries = []
        for img in images:
            try:
                # Resize image to reduce token usage
                image_bytes = resize_image(img['path'])

                # Convert to base64 for API
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                # Log the size reduction
                original_size = os.path.getsize(img['path'])
                resized_size = len(image_bytes)
                reduction_percent = (
                    (original_size - resized_size) / original_size) * 100 if original_size > 0 else 0
                model_logger.info(
                    f"Image resized from {original_size/1024:.1f}KB to {resized_size/1024:.1f}KB ({reduction_percent:.1f}% reduction)")

                # Get summary from OpenAI GPT-4o
                response = client.chat.completions.create(
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
                model_logger.info(
                    f"Generated summary for image on page {img['page']} using GPT-4o")

                # Create document
                doc = Document(
                    page_content=f"IMAGE: {summary}",
                    metadata={
                        'page': img['page'],
                        'type': 'image',
                        'image_path': img['path']
                    }
                )
                summaries.append(doc)

            except Exception as e:
                error_msg = f"Error generating summary for image {img['path']}: {str(e)}"
                model_logger.error(error_msg)
                error_logger.error(error_msg, exc_info=True)

        model_logger.info(f"Generated {len(summaries)} image summaries")
        return summaries


def index_document_to_faiss(file_path: str, file_id: int) -> bool:
    """Main indexing function with text and image processing"""
    with PerformanceTimer(model_logger, f"index_document:{os.path.basename(file_path)}"):
        try:
            # Create output directories
            image_dir = os.path.join(faiss_db_path, "extracted_images")
            os.makedirs(image_dir, exist_ok=True)
            model_logger.info(
                f"Starting indexing for document: {file_path} (ID: {file_id})")

            # Extract content
            model_logger.info(f"Extracting text from {file_path}")
            texts = extract_text_pdfplumber(file_path)
            model_logger.info(f"Extracting images from {file_path}")
            images = extract_images_pymupdf(file_path, image_dir)

            # Process text
            model_logger.info(f"Processing {len(texts)} text chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
            )

            text_docs = [
                Document(
                    page_content=text['content'],
                    metadata={
                        'page': text['page'],
                        'file_id': file_id,
                        'type': 'text',
                        'source': file_path
                    }
                ) for text in texts
            ]
            text_chunks = text_splitter.split_documents(text_docs)
            model_logger.info(f"Split text into {len(text_chunks)} chunks")

            # Process images
            model_logger.info(f"Processing {len(images)} images")
            image_summaries = get_image_summaries(images)
            for doc in image_summaries:
                doc.metadata.update({'file_id': file_id})
            model_logger.info(
                f"Generated {len(image_summaries)} image summaries")

            # Combine and store
            all_docs = text_chunks + image_summaries
            model_logger.info(
                f"Indexing {len(all_docs)} total documents to FAISS")

            if all_docs:
                # Add documents to the vector store
                vectorstore.add_documents(all_docs)

                # Save the updated index
                vectorstore.save_local(collection_path)

                # Store the documents for this file ID for potential deletion later
                file_id_mapping[file_id] = all_docs

                model_logger.info(
                    f"Successfully indexed document {file_path} (ID: {file_id})")
                return True

            model_logger.warning(
                f"No content extracted from document {file_path}")
            return False

        except Exception as e:
            error_msg = f"Indexing error for {file_path}: {str(e)}"
            model_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return False


def delete_doc_from_faiss(file_id: int) -> bool:
    """Delete documents by file_id by rebuilding the index without those documents"""
    with PerformanceTimer(model_logger, f"delete_from_faiss:{file_id}"):
        try:
            model_logger.info(f"Deleting document ID {file_id} from FAISS")

            # FAISS doesn't support direct deletion, so we need to:
            # 1. Get all documents from the current index
            # 2. Filter out documents with the specified file_id
            # 3. Create a new index with the remaining documents

            # Get all documents (this is a simplified approach - in a real app,
            # you might need to store document metadata separately)
            all_docs = []
            for doc_file_id, docs in file_id_mapping.items():
                if doc_file_id != file_id:
                    all_docs.extend(docs)

            if not all_docs:
                # If there are no documents left, initialize with an empty document
                all_docs = [
                    Document(page_content="Initialization document", metadata={"init": True})]

            # Create a new vector store with the filtered documents
            new_vectorstore = FAISS.from_documents(
                all_docs,
                embedding_function
            )

            # Save the new index, replacing the old one
            new_vectorstore.save_local(collection_path)

            # Update the global vectorstore reference
            global vectorstore
            vectorstore = new_vectorstore

            # Remove the file_id from our mapping
            if file_id in file_id_mapping:
                del file_id_mapping[file_id]

            model_logger.info(
                f"Successfully deleted document ID {file_id} from FAISS")
            return True

        except Exception as e:
            error_msg = f"Deletion error for document ID {file_id}: {str(e)}"
            model_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            return False


def clean_faiss_db_except_current(current_file_id: int, clean_db: bool = False) -> bool:
    """
    Clean up the FAISS database and only keep the current document.

    Args:
        current_file_id (int): The ID of the document to keep.
        clean_db (bool): Whether to also clean up database records.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        model_logger.info(
            f"Cleaning FAISS DB except for document ID: {current_file_id}")

        # Get all document IDs in the database
        all_ids = list(file_id_mapping.keys())

        # Remove all documents except the current one
        for file_id in all_ids:
            if file_id != current_file_id:
                model_logger.info(f"Removing document ID: {file_id}")
                delete_doc_from_faiss(file_id)

                # If clean_db is True, also delete from database
                if clean_db:
                    from db_utils import delete_document_record
                    model_logger.info(
                        f"Removing document ID {file_id} from database")
                    delete_document_record(file_id)

        model_logger.info(
            f"FAISS DB cleaned, only document ID {current_file_id} remains")
        return True
    except Exception as e:
        error_logger.error(f"Error cleaning FAISS DB: {str(e)}", exc_info=True)
        return False
