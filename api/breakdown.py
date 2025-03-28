import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from logger import api_logger, error_logger, PerformanceTimer
from db_utils import get_document_path
from faiss_utils import extract_text_pdfplumber, extract_images_pymupdf, get_image_summaries
import tempfile
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def analyze_document(file_id: int, model: str = "gemini-2.0-flash") -> Dict[str, Any]:
    """
    Analyze a document and generate a structured breakdown.

    Args:
        file_id (int): The ID of the document to analyze.
        model (str): The Gemini model to use for analysis.

    Returns:
        Dict[str, Any]: The structured document breakdown.
    """
    with PerformanceTimer(api_logger, f"document_breakdown:{model}"):
        try:
            # Get document path from database
            document_path = get_document_path(file_id)
            if not document_path:
                error_msg = f"Document with ID {file_id} not found"
                error_logger.error(error_msg)
                return {"error": error_msg}

            # Check if file exists and is accessible
            if not os.path.exists(document_path):
                error_msg = f"Document file does not exist at path: {document_path}"
                error_logger.error(error_msg)
                return {"error": error_msg}

            # Log file details
            file_size = os.path.getsize(document_path)
            error_logger.info(f"Document file size: {file_size} bytes")

            # Check file extension
            _, file_extension = os.path.splitext(document_path)
            if file_extension.lower() != '.pdf':
                error_msg = f"Document is not a PDF file. Found extension: {file_extension}"
                error_logger.error(error_msg)
                return {"error": error_msg}

            # Extract text from document
            api_logger.info(f"Extracting text from document: {document_path}")
            try:
                text_content = extract_text_pdfplumber(document_path)
                if not text_content:
                    error_msg = f"No text content could be extracted from document: {document_path}"
                    error_logger.error(error_msg)
                    return {"error": error_msg}

                document_text = "\n\n".join(
                    [page["text"] for page in text_content])
                if not document_text.strip():
                    error_msg = f"Extracted text is empty for document: {document_path}"
                    error_logger.error(error_msg)
                    return {"error": error_msg}

                error_logger.info(
                    f"Successfully extracted {len(text_content)} pages of text")
            except Exception as text_error:
                error_msg = f"Error extracting text from document: {str(text_error)}"
                error_logger.error(error_msg, exc_info=True)
                return {"error": error_msg}

            # Extract and analyze images
            api_logger.info(
                f"Extracting images from document: {document_path}")
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    images = extract_images_pymupdf(document_path, temp_dir)
                    error_logger.info(
                        f"Extracted {len(images)} images from document")

                    image_summaries = get_image_summaries(images)
                    image_summary_text = "\n\n".join([f"Image {i+1}: {summary.page_content}"
                                                      for i, summary in enumerate(image_summaries)])
                    error_logger.info(
                        f"Generated summaries for {len(image_summaries)} images")
            except Exception as img_error:
                error_msg = f"Error processing images from document: {str(img_error)}"
                error_logger.error(error_msg, exc_info=True)
                # Continue with text-only analysis
                image_summary_text = "No images could be processed from the document."

            # Create prompt for Gemini
            prompt = create_breakdown_prompt(document_text, image_summary_text)

            # Call Gemini API
            api_logger.info(f"Calling Gemini API with model: {model}")
            try:
                response = call_gemini_api(prompt, model)
                if not response:
                    error_msg = "Empty response from Gemini API"
                    error_logger.error(error_msg)
                    return {"error": error_msg}
            except Exception as api_error:
                error_msg = f"Error calling Gemini API: {str(api_error)}"
                error_logger.error(error_msg, exc_info=True)
                return {"error": error_msg}

            # Parse and validate response
            api_logger.info("Parsing Gemini API response")
            try:
                breakdown = parse_gemini_response(response)
                if not breakdown:
                    error_msg = "Failed to parse Gemini API response"
                    error_logger.error(error_msg)
                    return {"error": error_msg}
            except Exception as parse_error:
                error_msg = f"Error parsing Gemini API response: {str(parse_error)}"
                error_logger.error(error_msg, exc_info=True)
                return {"error": error_msg}

            return breakdown

        except Exception as e:
            error_msg = f"Error analyzing document: {str(e)}"
            error_logger.error(error_msg, exc_info=True)
            return {"error": error_msg}


def create_breakdown_prompt(document_text: str, image_summaries: str) -> str:
    """
    Create a prompt for the Gemini API to analyze a document.

    Args:
        document_text (str): The text content of the document.
        image_summaries (str): Summaries of images in the document.

    Returns:
        str: The prompt for the Gemini API.
    """
    return f"""
You are an expert system architect and technical documentation analyzer. Analyze the uploaded document and provide a structured breakdown in the following format. Consider both text content and extracted image summaries to generate comprehensive insights.

IMPORTANT: Your response MUST be a valid JSON object with the exact structure specified below. Do not include any explanatory text, markdown formatting, or code blocks outside the JSON. Just return the raw JSON object.

The JSON structure must be:
{{
  "document_breakdown": {{
    "major_components": [
      {{
        "name": "Component Name",
        "description": "2-3 sentence explanation",
        "key_functions": ["list", "of", "capabilities"]
      }}
    ],
    "diagrams": [
      {{
        "type": "Diagram Type",
        "purpose": "Primary objective",
        "key_elements": ["list", "of", "elements"],
        "relation_to_system": "How it relates to the overall system"
      }}
    ],
    "api_contracts": [
      {{
        "endpoint": "/api/endpoint",
        "method": "HTTP Method",
        "parameters": [
          {{
            "name": "Parameter Name",
            "type": "Parameter Type",
            "description": "Parameter Description"
          }}
        ],
        "success_response": "Example success response",
        "error_codes": ["list", "of", "error", "codes"]
      }}
    ],
    "pii_data": {{
      "identified_fields": ["list", "of", "PII", "fields"],
      "handling_procedures": "Description of how PII is handled",
      "compliance_standards": ["list", "of", "compliance", "standards"]
    }}
  }}
}}

If you cannot find information for a specific field, use "Not explicitly specified" for text fields or empty arrays [] for list fields.

Document Text:
{document_text}

Image Summaries:
{image_summaries}

Remember, your response must be a valid JSON object with the exact structure specified above. Do not include any explanatory text, markdown formatting, or code blocks outside the JSON. Just return the raw JSON object.
"""


def call_gemini_api(prompt: str, model: str) -> str:
    """
    Call the Gemini API with the given prompt.

    Args:
        prompt (str): The prompt for the Gemini API.
        model (str): The Gemini model to use.

    Returns:
        str: The response from the Gemini API.
    """
    try:
        # Log the full prompt being sent to Gemini
        error_logger.info(f"Full prompt being sent to Gemini API: {prompt}")

        generation_config = {
            "temperature": 0.1,  # Lower temperature for more deterministic output
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",  # Request JSON response
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        response = model_instance.generate_content(prompt)

        # Log the full response from Gemini
        error_logger.info(f"Full response from Gemini API: {str(response)}")

        # Check if the response has parts and if the first part has text
        if hasattr(response, 'parts') and len(response.parts) > 0:
            response_text = response.parts[0].text
            error_logger.info(
                f"Extracted text from response: {response_text[:500]}...")
            return response_text

        # Fall back to the text attribute if parts is not available
        response_text = response.text
        error_logger.info(
            f"Extracted text from response.text: {response_text[:500]}...")
        return response_text
    except Exception as e:
        error_logger.error(
            f"Error calling Gemini API: {str(e)}", exc_info=True)
        raise


def parse_gemini_response(response: str) -> Dict[str, Any]:
    """
    Parse the response from the Gemini API.

    Args:
        response (str): The response from the Gemini API.

    Returns:
        Dict[str, Any]: The parsed response.
    """
    try:
        error_logger.info(
            f"Starting to parse Gemini API response. First 200 chars: {response[:200]}...")

        # First, try to find JSON in code blocks (markdown format)
        json_block_pattern = r"```(?:json)?\s*\n([\s\S]*?)\n```"
        import re
        json_blocks = re.findall(json_block_pattern, response)

        if json_blocks:
            error_logger.info(
                f"Found {len(json_blocks)} JSON blocks in the response")
            # Try each JSON block until one works
            for i, json_block in enumerate(json_blocks):
                try:
                    error_logger.info(
                        f"Trying to parse JSON block {i+1}: {json_block[:200]}...")
                    data = json.loads(json_block.strip())
                    if "document_breakdown" in data:
                        error_logger.info(
                            "Successfully parsed JSON from code block with document_breakdown key")
                        breakdown = data["document_breakdown"]
                        break
                    else:
                        error_logger.info(
                            f"JSON block {i+1} does not contain document_breakdown key")
                except json.JSONDecodeError as e:
                    error_logger.info(
                        f"Failed to parse JSON block {i+1}: {str(e)}")
                    continue
            else:  # No valid JSON found in code blocks
                error_logger.info(
                    "No valid JSON found in code blocks, falling back to original method")
                # Fall back to the original method
                json_start = response.find("{")
                json_end = response.rfind("}")

                if json_start == -1 or json_end == -1:
                    error_logger.error("No JSON found in Gemini API response")
                    return {"error": "Invalid response format from Gemini API"}

                json_str = response[json_start:json_end+1]
                error_logger.info(
                    f"Extracted JSON string: {json_str[:200]}...")
                try:
                    data = json.loads(json_str)
                    if "document_breakdown" in data:
                        error_logger.info(
                            "Successfully parsed JSON with document_breakdown key")
                        breakdown = data["document_breakdown"]
                    else:
                        # Try to see if the JSON itself is the breakdown
                        error_logger.info(
                            "Checking if JSON itself is the breakdown")
                        if all(field in data for field in ["major_components", "diagrams", "api_contracts", "pii_data"]):
                            error_logger.info(
                                "JSON contains all required fields, using as breakdown")
                            breakdown = data
                        else:
                            error_logger.error(
                                "Missing 'document_breakdown' in Gemini API response")
                            return {"error": "Invalid response structure from Gemini API"}
                except json.JSONDecodeError as e:
                    error_logger.error(f"Error parsing JSON: {str(e)}")
                    return {"error": f"Error parsing JSON from Gemini API response: {str(e)}"}
        else:
            error_logger.info(
                "No JSON code blocks found, trying original method")
            # No code blocks found, try the original method
            json_start = response.find("{")
            json_end = response.rfind("}")

            if json_start == -1 or json_end == -1:
                error_logger.error("No JSON found in Gemini API response")
                return {"error": "Invalid response format from Gemini API"}

            json_str = response[json_start:json_end+1]
            error_logger.info(f"Extracted JSON string: {json_str[:200]}...")
            try:
                data = json.loads(json_str)
                if "document_breakdown" in data:
                    error_logger.info(
                        "Successfully parsed JSON with document_breakdown key")
                    breakdown = data["document_breakdown"]
                else:
                    # Try to see if the JSON itself is the breakdown
                    error_logger.info(
                        "Checking if JSON itself is the breakdown")
                    if all(field in data for field in ["major_components", "diagrams", "api_contracts", "pii_data"]):
                        error_logger.info(
                            "JSON contains all required fields, using as breakdown")
                        breakdown = data
                    else:
                        error_logger.error(
                            "Missing 'document_breakdown' in Gemini API response")
                        return {"error": "Invalid response structure from Gemini API"}
            except json.JSONDecodeError as e:
                error_logger.error(f"Error parsing JSON: {str(e)}")
                return {"error": f"Error parsing JSON from Gemini API response: {str(e)}"}

        # Ensure all required fields are present
        required_fields = ["major_components",
                           "diagrams", "api_contracts", "pii_data"]
        for field in required_fields:
            if field not in breakdown:
                breakdown[field] = []
                if field == "pii_data":
                    breakdown[field] = {
                        "identified_fields": [],
                        "handling_procedures": "Not explicitly specified",
                        "compliance_standards": []
                    }

        # Validate the structure of each component
        if isinstance(breakdown["major_components"], list):
            for i, component in enumerate(breakdown["major_components"]):
                if not isinstance(component, dict):
                    breakdown["major_components"][i] = {
                        "name": str(component),
                        "description": "No description provided",
                        "key_functions": []
                    }
                else:
                    if "name" not in component:
                        component["name"] = f"Component {i+1}"
                    if "description" not in component:
                        component["description"] = "No description provided"
                    if "key_functions" not in component:
                        component["key_functions"] = []
                    elif not isinstance(component["key_functions"], list):
                        component["key_functions"] = [
                            str(component["key_functions"])]

        if isinstance(breakdown["diagrams"], list):
            for i, diagram in enumerate(breakdown["diagrams"]):
                if not isinstance(diagram, dict):
                    breakdown["diagrams"][i] = {
                        "type": str(diagram),
                        "purpose": "No purpose provided",
                        "key_elements": [],
                        "relation_to_system": "Not specified"
                    }
                else:
                    if "type" not in diagram:
                        diagram["type"] = f"Diagram {i+1}"
                    if "purpose" not in diagram:
                        diagram["purpose"] = "No purpose provided"
                    if "key_elements" not in diagram:
                        diagram["key_elements"] = []
                    elif not isinstance(diagram["key_elements"], list):
                        diagram["key_elements"] = [
                            str(diagram["key_elements"])]
                    if "relation_to_system" not in diagram:
                        diagram["relation_to_system"] = "Not specified"

        if isinstance(breakdown["api_contracts"], list):
            for i, contract in enumerate(breakdown["api_contracts"]):
                if not isinstance(contract, dict):
                    breakdown["api_contracts"][i] = {
                        "endpoint": str(contract),
                        "method": "GET",
                        "parameters": [],
                        "success_response": "{}",
                        "error_codes": []
                    }
                else:
                    if "endpoint" not in contract:
                        contract["endpoint"] = f"/api/v1/endpoint{i+1}"
                    if "method" not in contract:
                        contract["method"] = "GET"
                    if "parameters" not in contract:
                        contract["parameters"] = []
                    elif not isinstance(contract["parameters"], list):
                        contract["parameters"] = []
                    if "success_response" not in contract:
                        contract["success_response"] = "{}"
                    if "error_codes" not in contract:
                        contract["error_codes"] = []
                    elif not isinstance(contract["error_codes"], list):
                        contract["error_codes"] = [
                            str(contract["error_codes"])]

        if isinstance(breakdown["pii_data"], dict):
            pii_data = breakdown["pii_data"]
            if "identified_fields" not in pii_data:
                pii_data["identified_fields"] = []
            elif not isinstance(pii_data["identified_fields"], list):
                pii_data["identified_fields"] = [
                    str(pii_data["identified_fields"])]

            if "handling_procedures" not in pii_data:
                pii_data["handling_procedures"] = "Not explicitly specified"

            if "compliance_standards" not in pii_data:
                pii_data["compliance_standards"] = []
            elif not isinstance(pii_data["compliance_standards"], list):
                pii_data["compliance_standards"] = [
                    str(pii_data["compliance_standards"])]

        return breakdown
    except Exception as e:
        error_logger.error(
            f"Error parsing Gemini API response: {str(e)}", exc_info=True)
        return {"error": f"Error parsing Gemini API response: {str(e)}"}
