#!/usr/bin/env python3
"""
Simple script to test the document breakdown functionality with a sample document.
This script can be run directly to analyze a sample document and print the results.
"""

from api.faiss_utils import extract_text_pdfplumber, extract_images_pymupdf, get_image_summaries
from api.breakdown import analyze_document, create_breakdown_prompt, call_gemini_api, parse_gemini_response
import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Load environment variables
load_dotenv()


def analyze_sample_document(doc_path, model="gemini-2.0-flash", output_file=None):
    """
    Analyze a sample document and print the results.

    Args:
        doc_path (str): Path to the document to analyze.
        model (str): The model to use for analysis.
        output_file (str): Path to save the results to (optional).
    """
    print(f"Analyzing document: {doc_path}")
    print(f"Using model: {model}")

    # Check if the document exists
    if not os.path.exists(doc_path):
        print(f"Error: Document not found at {doc_path}")
        return

    # Check if GEMINI_API_KEY is set
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set")
        return

    try:
        # Extract text from document
        print("Extracting text from document...")
        text_content = extract_text_pdfplumber(doc_path)
        document_text = "\n\n".join([page["text"] for page in text_content])
        print(f"Extracted {len(text_content)} pages of text")

        # Extract and analyze images
        print("Extracting images from document...")
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            images = extract_images_pymupdf(doc_path, temp_dir)
            print(f"Extracted {len(images)} images")

            if len(images) > 0:
                print("Generating summaries for images...")
                image_summaries = get_image_summaries(images)
                image_summary_text = "\n\n".join([f"Image {i+1}: {summary.page_content}"
                                                  for i, summary in enumerate(image_summaries)])
                print(f"Generated summaries for {len(image_summaries)} images")
            else:
                image_summary_text = "No images found in the document."

        # Create prompt for Gemini
        print("Creating prompt for Gemini API...")
        prompt = create_breakdown_prompt(document_text, image_summary_text)

        # Call Gemini API
        print(f"Calling Gemini API with model: {model}...")
        response = call_gemini_api(prompt, model)

        # Parse and validate response
        print("Parsing Gemini API response...")
        breakdown = parse_gemini_response(response)

        # Check if there was an error
        if "error" in breakdown:
            print(f"Error: {breakdown['error']}")
            return

        # Print summary of the breakdown
        print("\n=== Document Breakdown Summary ===")
        print(f"Major Components: {len(breakdown['major_components'])}")
        print(f"Diagrams: {len(breakdown['diagrams'])}")
        print(f"API Contracts: {len(breakdown['api_contracts'])}")
        print(f"PII Fields: {len(breakdown['pii_data']['identified_fields'])}")

        # Save results to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(breakdown, f, indent=2)
            print(f"\nResults saved to {output_file}")

        # Print detailed breakdown
        print("\n=== Detailed Breakdown ===")
        print(json.dumps(breakdown, indent=2))

    except Exception as e:
        print(f"Error analyzing document: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze a sample document and print the results.")
    parser.add_argument("--doc", type=str,
                        help="Path to the document to analyze")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Model to use for analysis (default: gemini-2.0-flash)")
    parser.add_argument("--output", type=str,
                        help="Path to save the results to (optional)")

    args = parser.parse_args()

    # If no document is specified, use the default test document
    if not args.doc:
        # Use the document in the root directory
        doc_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "doc-test.pdf")
        print(f"No document specified, using default: {doc_path}")
    else:
        doc_path = args.doc

    analyze_sample_document(doc_path, args.model, args.output)


if __name__ == "__main__":
    main()
