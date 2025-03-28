#!/usr/bin/env python3
"""
Test script for document breakdown functionality.
This script tests the document analysis and breakdown features.
"""

from api.faiss_utils import index_document_to_faiss, extract_text_pdfplumber, extract_images_pymupdf, get_image_summaries
from api.db_utils import get_document_path, insert_document_record
from api.breakdown import analyze_document, create_breakdown_prompt, call_gemini_api, parse_gemini_response
import logging
import os
import sys
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("document_breakdown_test")

# Path to test documents
TEST_DOCS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DOC_PATH = os.path.join(TEST_DOCS_DIR, "doc-test.pdf")
TEST_IMAGE_DOC_PATH = os.path.join(TEST_DOCS_DIR, "image-based-pdf-sample.pdf")


class TestDocumentBreakdown(unittest.TestCase):
    """Test cases for document breakdown functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Ensure the document collection directory exists
        self.collection_dir = os.path.join(
            self.temp_dir, "document_collection")
        os.makedirs(self.collection_dir, exist_ok=True)

        # Copy test documents to the temporary directory
        self.test_doc = os.path.join(self.temp_dir, "doc-test.pdf")
        shutil.copy(TEST_DOC_PATH, self.test_doc)

        self.test_image_doc = os.path.join(
            self.temp_dir, "image-based-pdf-sample.pdf")
        shutil.copy(TEST_IMAGE_DOC_PATH, self.test_image_doc)

        logger.info(
            f"Test environment set up with documents in {self.temp_dir}")

    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
        logger.info("Test environment cleaned up")

    def test_extract_text(self):
        """Test text extraction from PDF."""
        logger.info("Testing text extraction from PDF")

        # Extract text from test document
        text_content = extract_text_pdfplumber(self.test_doc)

        # Verify text extraction
        self.assertIsNotNone(text_content)
        self.assertGreater(len(text_content), 0)
        self.assertIsInstance(text_content, list)

        # Check content of first page
        self.assertIn("text", text_content[0])
        self.assertGreater(len(text_content[0]["text"]), 0)

        logger.info(f"Successfully extracted text from {self.test_doc}")

    def test_extract_images(self):
        """Test image extraction from PDF."""
        logger.info("Testing image extraction from PDF")

        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as img_dir:
            # Extract images from test document
            images = extract_images_pymupdf(self.test_image_doc, img_dir)

            # Verify image extraction
            self.assertIsNotNone(images)
            self.assertIsInstance(images, list)

            # Check if any images were extracted (the test document should have images)
            if len(images) > 0:
                logger.info(
                    f"Successfully extracted {len(images)} images from {self.test_image_doc}")
            else:
                logger.warning(f"No images found in {self.test_image_doc}")

    def test_image_summaries(self):
        """Test generating summaries for images."""
        logger.info("Testing image summary generation")

        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as img_dir:
            # Extract images from test document
            images = extract_images_pymupdf(self.test_image_doc, img_dir)

            # Skip test if no images were found
            if len(images) == 0:
                logger.warning(
                    "Skipping image summary test as no images were found")
                return

            # Generate summaries for images
            summaries = get_image_summaries(images)

            # Verify summaries
            self.assertIsNotNone(summaries)
            self.assertIsInstance(summaries, list)
            self.assertGreater(len(summaries), 0)

            # Check content of first summary
            self.assertIsNotNone(summaries[0].page_content)
            self.assertGreater(len(summaries[0].page_content), 0)

            logger.info(
                f"Successfully generated summaries for {len(summaries)} images")

    def test_create_breakdown_prompt(self):
        """Test creating a prompt for document breakdown."""
        logger.info("Testing breakdown prompt creation")

        # Create a sample prompt
        document_text = "This is a sample document text for testing."
        image_summaries = "This is a sample image summary for testing."

        # Generate prompt
        prompt = create_breakdown_prompt(document_text, image_summaries)

        # Verify prompt
        self.assertIsNotNone(prompt)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

        # Check if prompt contains the document text and image summaries
        self.assertIn(document_text, prompt)
        self.assertIn(image_summaries, prompt)

        logger.info("Successfully created breakdown prompt")

    def test_call_gemini_api(self):
        """Test calling the Gemini API."""
        logger.info("Testing Gemini API call")

        # Skip test if GEMINI_API_KEY is not set
        if not os.getenv("GEMINI_API_KEY"):
            logger.warning(
                "Skipping Gemini API test as GEMINI_API_KEY is not set")
            return

        # Create a simple prompt
        prompt = "Generate a short JSON response with the following structure: {\"test\": \"success\"}"

        try:
            # Call Gemini API
            response = call_gemini_api(prompt, "gemini-2.0-flash")

            # Verify response
            self.assertIsNotNone(response)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

            logger.info("Successfully called Gemini API")
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            self.fail(f"Gemini API call failed: {str(e)}")

    def test_parse_gemini_response(self):
        """Test parsing Gemini API response."""
        logger.info("Testing Gemini response parsing")

        # Sample response with the expected structure
        sample_response = """
        Here's the document breakdown:

        ```json
        {
          "document_breakdown": {
            "major_components": [
              {
                "name": "Authentication Service",
                "description": "Handles user authentication and authorization.",
                "key_functions": ["Login", "Logout", "Token Validation"]
              }
            ],
            "diagrams": [
              {
                "type": "System Architecture",
                "purpose": "Shows system components and their relationships",
                "key_elements": ["Frontend", "Backend", "Database"],
                "relation_to_system": "Provides overview of system architecture"
              }
            ],
            "api_contracts": [
              {
                "endpoint": "/api/v1/auth",
                "method": "POST",
                "parameters": [
                  {
                    "name": "username",
                    "type": "string",
                    "description": "User's username"
                  }
                ],
                "success_response": "{ \"token\": \"jwt_token\" }",
                "error_codes": ["401", "500"]
              }
            ],
            "pii_data": {
              "identified_fields": ["username", "email"],
              "handling_procedures": "Data is encrypted at rest and in transit.",
              "compliance_standards": ["GDPR", "CCPA"]
            }
          }
        }
        ```
        """

        # Parse response
        result = parse_gemini_response(sample_response)

        # Verify result
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

        # Check if result contains the expected fields
        self.assertIn("major_components", result)
        self.assertIn("diagrams", result)
        self.assertIn("api_contracts", result)
        self.assertIn("pii_data", result)

        # Check content of major_components
        self.assertEqual(len(result["major_components"]), 1)
        self.assertEqual(result["major_components"][0]
                         ["name"], "Authentication Service")

        logger.info("Successfully parsed Gemini response")

    def test_end_to_end_breakdown(self):
        """Test end-to-end document breakdown process."""
        logger.info("Testing end-to-end document breakdown")

        # Skip test if GEMINI_API_KEY is not set
        if not os.getenv("GEMINI_API_KEY"):
            logger.warning(
                "Skipping end-to-end test as GEMINI_API_KEY is not set")
            return

        try:
            # Mock the get_document_path function to return our test document
            def mock_get_document_path(file_id):
                return self.test_doc

            # Store the original function
            original_get_document_path = sys.modules["api.breakdown"].get_document_path

            # Replace with mock function
            sys.modules["api.breakdown"].get_document_path = mock_get_document_path

            # Call analyze_document
            result = analyze_document(1, "gemini-2.0-flash")

            # Restore original function
            sys.modules["api.breakdown"].get_document_path = original_get_document_path

            # Verify result
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)

            # Check if result contains the expected fields
            if "error" in result:
                logger.error(f"Error in document breakdown: {result['error']}")
                self.fail(f"Document breakdown failed: {result['error']}")
            else:
                self.assertIn("major_components", result)
                self.assertIn("diagrams", result)
                self.assertIn("api_contracts", result)
                self.assertIn("pii_data", result)

                logger.info(
                    "Successfully performed end-to-end document breakdown")

                # Print a summary of the breakdown
                logger.info(
                    f"Found {len(result['major_components'])} major components")
                logger.info(f"Found {len(result['diagrams'])} diagrams")
                logger.info(
                    f"Found {len(result['api_contracts'])} API contracts")
                logger.info(
                    f"Found {len(result['pii_data']['identified_fields'])} PII fields")

        except Exception as e:
            logger.error(f"Error in end-to-end test: {str(e)}")
            self.fail(f"End-to-end test failed: {str(e)}")


if __name__ == "__main__":
    unittest.main()
