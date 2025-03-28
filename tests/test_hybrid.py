import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image  # Add PIL import for image resizing
import fitz  # PyMuPDF
import pdfplumber
import tempfile
import shutil
import time
import traceback

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def test_gemini_text_generation():
    """Test basic text generation with Gemini."""
    try:
        # Create a Gemini model instance
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Generate content
        response = model.generate_content(
            "Explain how AI works in one paragraph.")

        print("\n=== Gemini Text Generation Test ===")
        print(response.text)

        return True
    except Exception as e:
        print(f"Error in Gemini text generation test: {str(e)}")
        return False


def test_gemini_embeddings():
    """Test embeddings with Gemini."""
    try:
        # Initialize embeddings
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            task_type="retrieval_document"
        )

        # Get embeddings for a text
        text = "This is a test document for embeddings."
        embedding = embedding_model.embed_query(text)

        print("\n=== Gemini Embeddings Test ===")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

        # Test similarity between two texts
        text2 = "Another document for testing embeddings."
        embedding2 = embedding_model.embed_query(text2)

        # Calculate cosine similarity
        similarity = np.dot(embedding, embedding2) / \
            (np.linalg.norm(embedding) * np.linalg.norm(embedding2))
        print(f"Similarity between two texts: {similarity}")

        return True
    except Exception as e:
        print(f"Error in Gemini embeddings test: {str(e)}")
        return False


def test_openai_vision():
    """Test vision capabilities with OpenAI."""
    try:
        # Use a sample image URL
        image_url = "https://storage.googleapis.com/generativeai-downloads/images/scones.jpg"

        # Download the image
        response = requests.get(image_url)
        image_bytes = response.content

        # Resize image to reduce token count
        img = Image.open(BytesIO(image_bytes))

        # Calculate new dimensions while maintaining aspect ratio
        max_dimension = 800  # Reasonable size that balances quality and token count
        width, height = img.size

        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to JPEG format with compression to further reduce size
        buffer = BytesIO()
        resized_img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        print(f"Original image size: {len(image_bytes)} bytes")
        print(f"Resized image size: {len(buffer.getvalue())} bytes")
        print(f"Base64 string length: {len(image_base64)} characters")

        # Generate content with the image using OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Describe the given image in detail."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ],
            max_tokens=300
        )

        print("\n=== OpenAI Vision Test ===")
        print(response.choices[0].message.content)

        return True
    except Exception as e:
        print(f"Error in OpenAI vision test: {str(e)}")
        traceback.print_exc()  # Add detailed error traceback
        return False


def test_langchain_integration():
    """Test LangChain integration with Gemini."""
    try:
        # Initialize LangChain Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7,
        )

        # Generate a response
        response = llm.invoke("What are the key principles of cybersecurity?")

        print("\n=== LangChain Integration Test ===")
        print(response.content)

        return True
    except Exception as e:
        print(f"Error in LangChain integration test: {str(e)}")
        return False


def test_pdf_image_processing():
    """Test PDF processing with image extraction and resizing."""
    try:
        # Use a sample PDF file from the project
        pdf_path = "../image-based-pdf-sample.pdf"
        if not os.path.exists(pdf_path):
            pdf_path = "image-based-pdf-sample.pdf"  # Try alternative path
            if not os.path.exists(pdf_path):
                print("Sample PDF file not found. Skipping PDF image processing test.")
                return False

        print(f"\n=== PDF Image Processing Test ===")
        print(f"Processing PDF: {pdf_path}")

        # Create a temporary directory for extracted images
        temp_dir = tempfile.mkdtemp()
        try:
            # Extract images from PDF
            pdf_document = fitz.open(pdf_path)
            image_count = 0
            processed_count = 0

            # Process only first 3 pages for testing
            for page_num in range(min(3, len(pdf_document))):
                page = pdf_document[page_num]
                image_list = page.get_images()

                print(f"Found {len(image_list)} images on page {page_num + 1}")
                image_count += len(image_list)

                for img_index, img in enumerate(image_list):
                    if img_index >= 2:  # Process only first 2 images per page for testing
                        continue

                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]

                        # Process the image - resize it
                        pil_img = Image.open(BytesIO(image_bytes))

                        # Calculate new dimensions while maintaining aspect ratio
                        max_dimension = 800
                        width, height = pil_img.size

                        if width > height:
                            new_width = max_dimension
                            new_height = int(height * (max_dimension / width))
                        else:
                            new_height = max_dimension
                            new_width = int(width * (max_dimension / height))

                        # Resize the image
                        resized_img = pil_img.resize(
                            (new_width, new_height), Image.LANCZOS)

                        # Convert to JPEG format with compression
                        buffer = BytesIO()
                        resized_img.save(buffer, format="JPEG", quality=85)
                        buffer.seek(0)

                        # Convert to base64
                        image_base64 = base64.b64encode(
                            buffer.getvalue()).decode('utf-8')

                        # Log size reduction
                        original_size = len(image_bytes)
                        new_size = len(buffer.getvalue())
                        reduction_percent = (
                            (original_size - new_size) / original_size) * 100
                        print(
                            f"Image {img_index} on page {page_num + 1}: Reduced from {original_size} to {new_size} bytes ({reduction_percent:.1f}% reduction)")

                        # Test with OpenAI vision API
                        if processed_count < 2:  # Only test first 2 images with OpenAI to save API calls
                            start_time = time.time()
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system",
                                        "content": "Describe the given image in detail."},
                                    {"role": "user",
                                        "content": [
                                            {"type": "text",
                                                "text": "Describe this image:"},
                                            {"type": "image_url", "image_url": {
                                                "url": f"data:image/jpeg;base64,{image_base64}"}}
                                        ]}
                                ],
                                max_tokens=300
                            )
                            elapsed_time = time.time() - start_time

                            print(
                                f"OpenAI vision API response time: {elapsed_time:.2f} seconds")
                            print(
                                f"Image description: {response.choices[0].message.content[:100]}...")

                        processed_count += 1

                    except Exception as e:
                        print(
                            f"Error processing image {img_index} from page {page_num + 1}: {e}")
                        traceback.print_exc()  # Add detailed error traceback

            pdf_document.close()
            print(
                f"Successfully processed {processed_count} out of {image_count} images")
            return processed_count > 0

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"Error in PDF image processing test: {str(e)}")
        traceback.print_exc()  # Add detailed error traceback
        return False


if __name__ == "__main__":
    print("Starting hybrid API tests (Gemini + OpenAI)...")

    # Run all tests
    gemini_text_test = test_gemini_text_generation()
    gemini_embeddings_test = test_gemini_embeddings()
    openai_vision_test = test_openai_vision()
    pdf_image_test = test_pdf_image_processing()
    langchain_test = test_langchain_integration()

    # Print summary
    print("\n=== Test Summary ===")
    print(
        f"Gemini Text Generation: {'✅ PASSED' if gemini_text_test else '❌ FAILED'}")
    print(
        f"Gemini Embeddings: {'✅ PASSED' if gemini_embeddings_test else '❌ FAILED'}")
    print(f"OpenAI Vision: {'✅ PASSED' if openai_vision_test else '❌ FAILED'}")
    print(
        f"PDF Image Processing: {'✅ PASSED' if pdf_image_test else '❌ FAILED'}")
    print(
        f"LangChain Integration: {'✅ PASSED' if langchain_test else '❌ FAILED'}")

    # Overall result
    if all([gemini_text_test, gemini_embeddings_test, openai_vision_test, pdf_image_test, langchain_test]):
        print("\n🎉 All tests passed! The hybrid API integration is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
