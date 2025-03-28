from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime
from typing import Optional, Annotated


class ModelName(str, Enum):
    # Gemini models
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_PRO = "gemini-2.0-pro"

    # No OpenAI models - removed


class QueryInput(BaseModel):
    session_id: Optional[str] = None
    question: str  # Mandatory field
    model: str = "gemini-2.0-flash"  # Changed from ModelName to str to accept any value
    # Whether to use hybrid search (vector + BM25) or just vector search
    use_hybrid_search: bool = True

    # Validator to ensure model is a valid Gemini model
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        # If not a Gemini model, default to gemini-2.0-flash
        if not v.startswith("gemini"):
            return "gemini-2.0-flash"
        # If it's already a valid Gemini model, return as is
        if v in [m.value for m in ModelName]:
            return v
        # If it's a Gemini model but not in our enum, default to gemini-2.0-flash
        return "gemini-2.0-flash"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "some-uuid-here",
                    "question": "What is RAG?",
                    "model": "gemini-2.0-flash",
                    "use_hybrid_search": True
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    answer: str
    processing_time: float
    model: str  # Changed from ModelName to str to match QueryInput

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "RAG stands for Retrieval Augmented Generation...",
                    "processing_time": 1.25,
                    "model": "gemini-2.0-flash"
                }
            ]
        }
    }


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    file_id: int


# New models for document breakdown

class ComponentInfo(BaseModel):
    name: str
    description: str
    key_functions: List[str]


class DiagramInfo(BaseModel):
    type: str
    purpose: str
    key_elements: List[str]
    relation_to_system: str


class ApiParameterInfo(BaseModel):
    name: str
    type: str
    description: str


class ApiContractInfo(BaseModel):
    endpoint: str
    method: str
    parameters: List[ApiParameterInfo]
    success_response: str
    error_codes: List[str]


class PiiDataInfo(BaseModel):
    identified_fields: List[str]
    handling_procedures: str
    compliance_standards: List[str]


class DocumentBreakdownResponse(BaseModel):
    major_components: List[ComponentInfo]
    diagrams: List[DiagramInfo]
    api_contracts: List[ApiContractInfo]
    pii_data: PiiDataInfo


class DocumentBreakdownRequest(BaseModel):
    file_id: int
    model: str = "gemini-2.0-flash"  # Using a more capable model for document analysis


# User and Authentication models
class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"


class UserCreate(BaseModel):
    username: str
    password: str
    role: UserRole = UserRole.USER


class UserModify(BaseModel):
    user_id: int
    new_username: str


class UserDelete(BaseModel):
    user_id: int


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    role: UserRole


class LoginResponse(BaseModel):
    user_id: int
    username: str
    role: UserRole
