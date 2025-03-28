#!/usr/bin/env python3
"""
Script to test the document breakdown API endpoint.
This script uploads a document and then calls the breakdown API endpoint.
"""

import os
import sys
import json
import argparse
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_breakdown_api(doc_path, api_url="http://localhost:8000", model="gemini-2.0-flash", output_file=None):
    """
    Test the document breakdown API endpoint.

    Args:
        doc_path (str): Path to the document to upload and analyze.
        api_url (str): Base URL of the API.
        model (str): The model to use for analysis.
        output_file (str): Path to save the results to (optional).
    """
    print(f"Testing document breakdown API with document: {doc_path}")
    print(f"API URL: {api_url}")
    print(f"Using model: {model}")

    # Check if the document exists
    if not os.path.exists(doc_path):
        print(f"Error: Document not found at {doc_path}")
        return

    try:
        # Step 1: Upload the document
        print("\n=== Step 1: Uploading document ===")

        with open(doc_path, 'rb') as f:
            files = {'file': (os.path.basename(doc_path), f)}
            upload_response = requests.post(
                f"{api_url}/upload-doc", files=files)

        if upload_response.status_code != 200:
            print(f"Error uploading document: {upload_response.text}")
            return

        upload_data = upload_response.json()
        file_id = upload_data.get('file_id')

        if not file_id:
            print("Error: No file ID returned from upload")
            return

        print(f"Document uploaded successfully with ID: {file_id}")

        # Step 2: Call the breakdown API
        print("\n=== Step 2: Calling document breakdown API ===")

        breakdown_data = {
            "file_id": file_id,
            "model": model
        }

        print(
            f"Sending request to {api_url}/document/analyze with data: {breakdown_data}")

        breakdown_response = requests.post(
            f"{api_url}/document/analyze", json=breakdown_data)

        if breakdown_response.status_code != 200:
            print(f"Error calling breakdown API: {breakdown_response.text}")
            return

        breakdown_result = breakdown_response.json()

        # Print summary of the breakdown
        print("\n=== Document Breakdown Summary ===")
        print(f"Major Components: {len(breakdown_result['major_components'])}")
        print(f"Diagrams: {len(breakdown_result['diagrams'])}")
        print(f"API Contracts: {len(breakdown_result['api_contracts'])}")
        print(
            f"PII Fields: {len(breakdown_result['pii_data']['identified_fields'])}")

        # Save results to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(breakdown_result, f, indent=2)
            print(f"\nResults saved to {output_file}")

        # Print detailed breakdown
        print("\n=== Detailed Breakdown ===")
        print(json.dumps(breakdown_result, indent=2))

        # Step 3: Clean up (optional) - delete the document
        print("\n=== Step 3: Cleaning up (deleting document) ===")

        delete_data = {
            "file_id": file_id
        }

        delete_response = requests.post(
            f"{api_url}/delete-doc", json=delete_data)

        if delete_response.status_code != 200:
            print(
                f"Warning: Failed to delete document: {delete_response.text}")
        else:
            print(f"Document with ID {file_id} deleted successfully")

    except Exception as e:
        print(f"Error testing breakdown API: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(
        description="Test the document breakdown API endpoint.")
    parser.add_argument("--doc", type=str,
                        help="Path to the document to upload and analyze")
    parser.add_argument("--api", type=str, default="http://localhost:8000",
                        help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Model to use for analysis (default: gemini-2.0-flash)")
    parser.add_argument("--output", type=str,
                        help="Path to save the results to (optional)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip cleanup (don't delete the document)")

    args = parser.parse_args()

    # If no document is specified, use the default test document
    if not args.doc:
        # Use the document in the root directory
        doc_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "doc-test.pdf")
        print(f"No document specified, using default: {doc_path}")
    else:
        doc_path = args.doc

    test_breakdown_api(doc_path, args.api, args.model, args.output)


if __name__ == "__main__":
    main()
