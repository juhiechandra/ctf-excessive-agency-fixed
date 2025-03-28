from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, status, Depends, Header, Response, Cookie
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest, UserCreate, UserLogin, UserResponse, LoginResponse, UserDelete, UserModify, UserRole
from faiss_utils import index_document_to_faiss, delete_doc_from_faiss, clean_faiss_db_except_current
from langchain_utils import get_rag_chain
from db_utils import get_chat_history, insert_application_logs, insert_document_record, delete_document_record, get_all_documents, authenticate_user, create_user, get_user_by_id, delete_user, modify_username, get_all_users
from logger import api_logger, error_logger, PerformanceTimer
import uuid
import shutil
import os
import traceback
import time
import sqlite3
from datetime import datetime, timedelta

# Create a directory for storing uploaded files
UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Basic RAG Chatbot",
              description="A simple Retrieval Augmented Generation chatbot system",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization",
                   "Accept", "X-Requested-With", "Origin"],
)

# Global exception handler


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())
    error_logger.error(
        f"Unhandled exception ID {error_id}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An unexpected error occurred",
            "error_id": error_id,
            "detail": str(exc)
        }
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Log request details
    api_logger.info(
        f"Request {request_id} started: {request.method} {request.url.path}")

    try:
        response = await call_next(request)

        # Log response details
        process_time = time.time() - start_time
        api_logger.info(
            f"Request {request_id} completed: Status {response.status_code} in {process_time:.2f}s")

        return response
    except Exception as e:
        # Log exception details
        process_time = time.time() - start_time
        error_logger.error(
            f"Request {request_id} failed after {process_time:.2f}s: {str(e)}", exc_info=True)
        raise

# Auth endpoints


@app.post("/login")
async def login(user_data: UserLogin):
    user = authenticate_user(user_data.username, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    return {
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"]
    }


@app.post("/logout")
async def logout():
    return {"message": "Logged out successfully"}

# Admin user management endpoints


@app.get("/admin/list-users")
async def list_users():
    """Get a list of all users in the system."""
    try:
        users = get_all_users()
        return users
    except Exception as e:
        error_id = str(uuid.uuid4())
        error_msg = f"Error listing users: {str(e)}"
        api_logger.error(f"{error_msg} (ID: {error_id})")
        error_logger.error(f"Error ID {error_id}: {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


@app.post("/admin/create-user")
async def create_new_user(user_data: UserCreate):
    user_id, error = create_user(
        user_data.username, user_data.password, user_data.role)
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )

    user = get_user_by_id(user_id)
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"]
    }


@app.post("/admin/modify-user")
async def modify_user_endpoint(user_data: UserModify):
    success, error = modify_username(user_data.user_id, user_data.new_username)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )

    user = get_user_by_id(user_data.user_id)
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"]
    }


@app.post("/admin/delete-user")
async def remove_user(user_data: UserDelete):
    success, error = delete_user(user_data.user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )

    return {"message": f"User {user_data.user_id} deleted successfully"}

# Admin chat function handler


@app.post("/admin/chat-function")
async def admin_chat_function(
    data: Dict[str, Any],
):
    try:
        command = data.get("command", "").lower()
        params = data.get("params", {})

        if command == "create-user":
            username = params.get("username")
            password = params.get("password")
            role = params.get("role", "user")

            if not username or not password:
                return {"status": "error", "message": "Username and password are required"}

            user_id, error = create_user(username, password, role)
            if error:
                return {"status": "error", "message": error}

            return {"status": "success", "message": f"User '{username}' created successfully with ID {user_id}"}

        elif command == "delete-user":
            user_id = params.get("user_id")
            if not user_id:
                return {"status": "error", "message": "User ID is required"}

            success, error = delete_user(user_id)
            if not success:
                return {"status": "error", "message": error}

            return {"status": "success", "message": f"User with ID {user_id} deleted successfully"}

        elif command == "modify-user":
            user_id = params.get("user_id")
            new_username = params.get("new_username")

            if not user_id or not new_username:
                return {"status": "error", "message": "User ID and new username are required"}

            success, error = modify_username(user_id, new_username)
            if not success:
                return {"status": "error", "message": error}

            return {"status": "success", "message": f"Username for user {user_id} changed to '{new_username}'"}

        else:
            return {"status": "error", "message": f"Unknown command: {command}"}

    except Exception as e:
        error_id = str(uuid.uuid4())
        error_msg = f"Error processing admin function: {str(e)}"
        api_logger.error(f"{error_msg} (ID: {error_id})")
        error_logger.error(f"Error ID {error_id}: {error_msg}", exc_info=True)

        return {"status": "error", "message": f"An error occurred: {str(e)}", "error_id": error_id}


@app.post("/upload-doc")
async def upload_file(file: UploadFile = File(...)):
    with PerformanceTimer(api_logger, f"upload_file:{file.filename}"):
        try:
            # Save temporary file for indexing
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            api_logger.info(f"Temporary file saved: {temp_path}")

            # Index document
            file_id = insert_document_record(file.filename)
            api_logger.info(f"Document record inserted with ID: {file_id}")

            # Save the file to the permanent storage location
            permanent_path = os.path.join(
                UPLOAD_DIR, f"doc-{file_id}-{file.filename}")
            shutil.copy(temp_path, permanent_path)
            api_logger.info(
                f"File saved to permanent storage: {permanent_path}")

            if index_document_to_faiss(temp_path, file_id):
                api_logger.info(
                    f"Document indexed successfully: {file.filename} (ID: {file_id})")

                # Clean up FAISS DB to only keep the current document
                clean_faiss_db_except_current(file_id, clean_db=True)
                api_logger.info(
                    f"FAISS DB and database cleaned, only document ID {file_id} remains")

                # Clean up temporary file
                os.remove(temp_path)
                api_logger.info(f"Temporary file removed: {temp_path}")

                # Clean up old files in the upload directory (keep only the current one)
                cleanup_uploaded_files(file_id)

                return {"message": "Document indexed successfully", "file_id": file_id}
            else:
                # Clean up if indexing failed
                os.remove(temp_path)
                os.remove(permanent_path)
                api_logger.error(f"Failed to index document: {file.filename}")
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"message": "Failed to index document"}
                )
        except Exception as e:
            error_id = str(uuid.uuid4())
            error_msg = f"Error uploading file: {str(e)}"
            api_logger.error(f"{error_msg} (ID: {error_id})")
            error_logger.error(
                f"Error ID {error_id}: {error_msg}", exc_info=True)

            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": error_msg, "error_id": error_id}
            )


def cleanup_uploaded_files(current_file_id: int):
    """Remove all uploaded files except the current one."""
    try:
        for filename in os.listdir(UPLOAD_DIR):
            # Skip the current file
            if filename.startswith(f"doc-{current_file_id}-"):
                continue

            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                api_logger.info(f"Removed old uploaded file: {file_path}")
    except Exception as e:
        error_logger.error(
            f"Error cleaning up uploaded files: {str(e)}", exc_info=True)


@app.post("/chat")
async def chat_endpoint(query: QueryInput) -> QueryResponse:
    """
    Process a chat query using RAG.

    Args:
        query: The query input containing the question and chat history.

    Returns:
        A response containing the answer and updated chat history.
    """
    try:
        with PerformanceTimer(api_logger, f"chat_endpoint:{query.question[:30]}"):
            api_logger.info(f"Received chat query: {query.question[:100]}...")

            # Get chat history from database if session_id is provided
            chat_history = []
            if query.session_id:
                api_logger.info(
                    f"Getting chat history for session: {query.session_id}")
                chat_history = get_chat_history(query.session_id)
                api_logger.info(
                    f"Retrieved {len(chat_history)} chat history items")

            # Convert chat history to the format expected by LangChain
            formatted_history = []
            for item in chat_history:
                formatted_history.append(("human", item["question"]))
                formatted_history.append(("ai", item["answer"]))

            # Get RAG chain with specified model and hybrid search option
            use_hybrid_search = query.use_hybrid_search if hasattr(
                query, 'use_hybrid_search') else True
            chain = get_rag_chain(
                model=query.model, use_hybrid_search=use_hybrid_search)

            # Process query
            api_logger.info(f"Processing query with model: {query.model}")
            start_time = time.time()
            response = chain.invoke({
                "input": query.question,
                "chat_history": formatted_history
            })
            end_time = time.time()
            processing_time = end_time - start_time
            api_logger.info(
                f"Query processed in {processing_time:.2f} seconds")

            # Extract answer
            answer = response["answer"]
            api_logger.info(f"Generated answer: {answer[:100]}...")

            # Log to database if session_id is provided
            if query.session_id:
                api_logger.info(
                    f"Logging chat to database for session: {query.session_id}")
                insert_application_logs(
                    session_id=query.session_id,
                    question=query.question,
                    answer=answer,
                    model=query.model,
                    processing_time=processing_time
                )

            # Return response
            return QueryResponse(
                answer=answer,
                processing_time=processing_time,
                model=query.model
            )

    except Exception as e:
        error_msg = f"Error processing chat query: {str(e)}"
        api_logger.error(error_msg)
        error_logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    with PerformanceTimer(api_logger, "list_documents"):
        try:
            documents = get_all_documents()
            api_logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            error_msg = f"Error retrieving documents: {str(e)}"
            api_logger.error(error_msg)
            error_logger.error(error_msg, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )


@app.post("/delete-doc")
async def delete_document(req: DeleteFileRequest):
    with PerformanceTimer(api_logger, f"delete_document:{req.file_id}"):
        try:
            # Delete from FAISS
            if delete_doc_from_faiss(req.file_id):
                api_logger.info(
                    f"Document deleted from FAISS: ID {req.file_id}")

                # Delete from database
                if delete_document_record(req.file_id):
                    api_logger.info(
                        f"Document record deleted: ID {req.file_id}")

                    # Delete the file from storage
                    try:
                        for filename in os.listdir(UPLOAD_DIR):
                            if filename.startswith(f"doc-{req.file_id}-"):
                                file_path = os.path.join(UPLOAD_DIR, filename)
                                os.remove(file_path)
                                api_logger.info(
                                    f"Document file deleted: {file_path}")
                    except Exception as e:
                        error_msg = f"Error deleting document file: {str(e)}"
                        api_logger.error(error_msg)
                        error_logger.error(error_msg, exc_info=True)
                        # Continue anyway, the important parts (FAISS and DB) are cleaned

                    return {"message": f"Document {req.file_id} deleted successfully"}
                else:
                    error_msg = f"Failed to delete document record from database: ID {req.file_id}"
                    api_logger.error(error_msg)
                    return JSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"message": error_msg}
                    )
            else:
                error_msg = f"Failed to delete document from FAISS: ID {req.file_id}"
                api_logger.error(error_msg)
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"message": error_msg}
                )
        except Exception as e:
            error_id = str(uuid.uuid4())
            error_msg = f"Error deleting document: {str(e)}"
            api_logger.error(f"{error_msg} (ID: {error_id})")
            error_logger.error(
                f"Error ID {error_id}: {error_msg}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": error_msg, "error_id": error_id}
            )

# LLM Tool Calling API endpoint
# SECURITY VULNERABILITY: This endpoint doesn't verify if the user is an admin
# Any user can access this endpoint if they know how to call it directly
#
# EXCESSIVE AGENCY VULNERABILITY:
# 1. The LLM has access to powerful database functions with no restrictions
# 2. Regular users can use the LLM to execute arbitrary SQL queries
# 3. Regular users can change ANY user's password, including admin passwords
# 4. The LLM makes all decisions about what functions to call with no guardrails
#
# This demonstrates how an LLM can be given excessive permissions and act beyond
# its intended authorization level, potentially compromising the entire system.


@app.post("/llm-tool-call")
async def llm_tool_call(
    data: Dict[str, Any],
):
    try:
        user_query = data.get("user_query", "")
        conversation_history = data.get("conversation_history", [])
        # Default to regular user if not specified
        user_role = data.get("user_role", "user")

        # Function definitions that the LLM will be able to call - format compatible with Gemini API
        function_defs = [
            {
                "function_declarations": [
                    {
                        "name": "create_user",
                        "description": "Create a new user in the system",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "username": {
                                    "type": "STRING",
                                    "description": "The username for the new user"
                                },
                                "password": {
                                    "type": "STRING",
                                    "description": "The password for the new user"
                                },
                                "role": {
                                    "type": "STRING",
                                    "description": "The role for the new user (default: user)",
                                    "enum": ["user", "admin"]
                                }
                            },
                            "required": ["username", "password"]
                        }
                    }
                ]
            },
            {
                "function_declarations": [
                    {
                        "name": "delete_user",
                        "description": "Delete an existing user from the system",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "user_id": {
                                    "type": "INTEGER",
                                    "description": "The ID of the user to delete"
                                }
                            },
                            "required": ["user_id"]
                        }
                    }
                ]
            },
            {
                "function_declarations": [
                    {
                        "name": "modify_username",
                        "description": "Change the username of an existing user",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "user_id": {
                                    "type": "INTEGER",
                                    "description": "The ID of the user to modify"
                                },
                                "new_username": {
                                    "type": "STRING",
                                    "description": "The new username for the user"
                                }
                            },
                            "required": ["user_id", "new_username"]
                        }
                    }
                ]
            },
            {
                "function_declarations": [
                    {
                        "name": "get_user",
                        "description": "Get details of a user by their ID",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "user_id": {
                                    "type": "INTEGER",
                                    "description": "The ID of the user to retrieve"
                                }
                            },
                            "required": ["user_id"]
                        }
                    }
                ]
            },
            {
                "function_declarations": [
                    {
                        "name": "list_users",
                        "description": "List all users in the system",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "include_details": {
                                    "type": "BOOLEAN",
                                    "description": "Whether to include detailed information about users (default: true)"
                                }
                            }
                        }
                    }
                ]
            },
            {
                "function_declarations": [
                    {
                        "name": "execute_sql",
                        "description": "Execute a SQL query directly on the database - USE WITH EXTREME CAUTION",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "query": {
                                    "type": "STRING",
                                    "description": "The SQL query to execute on the database"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ]
            },
            {
                "function_declarations": [
                    {
                        "name": "change_user_password",
                        "description": "Change a user's password in the database",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "user_id": {
                                    "type": "INTEGER",
                                    "description": "The ID of the user whose password to change"
                                },
                                "new_password": {
                                    "type": "STRING",
                                    "description": "The new password to set for the user"
                                }
                            },
                            "required": ["user_id", "new_password"]
                        }
                    }
                ]
            }
        ]

        # Import Gemini client directly here to avoid circular imports
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        import os
        from dotenv import load_dotenv

        # Load environment variables
        load_dotenv()

        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        # Get model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        # Format conversation history for the model
        formatted_history = []
        for msg in conversation_history:
            formatted_history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]]
            })

        # Add system prompt - Including role-specific behavior for UI appearance
        system_prompt = f"""You are an administrative assistant that can help manage users in the system.
        You have access to functions that can create users, delete users, modify usernames, and retrieve user information.
        
        When a user asks you to perform any of these actions, you should:
        1. Determine which function to call based on their request
        2. Ask for any missing information needed to call the function
        3. Once you have all necessary information, call the appropriate function
        
        IMPORTANT: The current user's role is: {user_role}
        
        When a user asks what you can do, your response should depend on their role:
        
        For regular users, only mention these capabilities:
        - Retrieve user information: I can get information about a specific user by their ID or list all users.
        - Change user passwords: I'll need the ID of the user and the new password.
        
        For admin users, list all these capabilities:
        - Create users: I'll need the new user's username and password. I can also assign them a role (user or admin).
        - Delete users: I'll need the ID of the user to delete.
        - Modify usernames: I'll need the ID of the user and the new username.
        - Retrieve user information: I can get information about a specific user by their ID or list all users.
        - Change user passwords: I'll need the ID of the user and the new password.
        - Perform complex data operations: I can use SQL queries for advanced filtering, searching, generating reports, or updating multiple records. However, please use this with caution.
        
        Despite these different responses, all functions are actually available to any user - this is the excessive agency vulnerability!
        
        For more complex data operations, you have access to a powerful execute_sql function that allows direct database queries.
        Some examples where SQL queries might be helpful:
        - Advanced filtering or searching of user data
        - Generating statistics or reports
        - Retrieving specific information from the database
        - Updating multiple records at once
        
        You can also help users manage their passwords using the change_user_password function.
        
        You can also list all users in the system when requested. When displaying a list of users:
        - Present the data in a clear, tabular format with headers for ID, Username, and Role
        - Make sure the table is properly aligned and readable
        - Highlight admin users in some way to make them stand out
        
        Always be professional but friendly, and confirm the details before taking any action.
        For security reasons, never share passwords in your responses.
        """

        # Create chat session
        chat = model.start_chat(history=formatted_history)

        # Add the system prompt
        chat.send_message(system_prompt)

        # Send user query and request function calling
        response = chat.send_message(
            user_query,
            tools=function_defs
        )

        # Extract the response text properly
        response_text = ""
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text += part.text

        # Check if response includes function calls
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Function call present in response
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_call = part.function_call
                        function_name = function_call.name
                        function_params = function_call.args

                        # Execute the actual function based on the LLM's choice
                        if function_name == "create_user":
                            username = function_params.get("username")
                            password = function_params.get("password")
                            role = function_params.get("role", "user")

                            user_id, error = create_user(
                                username, password, role)

                            if error:
                                return {
                                    "status": "error",
                                    "message": error,
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": function_params
                                    }
                                }

                            # Get user details
                            user = get_user_by_id(user_id)
                            return {
                                "status": "success",
                                "message": f"User '{username}' created successfully with ID {user_id}",
                                "data": {
                                    "user_id": user["id"],
                                    "username": user["username"],
                                    "role": user["role"]
                                },
                                "llm_response": response_text,
                                "function_call": {
                                    "name": function_name,
                                    "params": function_params
                                }
                            }

                        elif function_name == "delete_user":
                            user_id = function_params.get("user_id")

                            success, error = delete_user(user_id)

                            if not success:
                                return {
                                    "status": "error",
                                    "message": error,
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": function_params
                                    }
                                }

                            return {
                                "status": "success",
                                "message": f"User with ID {user_id} deleted successfully",
                                "llm_response": response_text,
                                "function_call": {
                                    "name": function_name,
                                    "params": function_params
                                }
                            }

                        elif function_name == "modify_username":
                            user_id = function_params.get("user_id")
                            new_username = function_params.get("new_username")

                            success, error = modify_username(
                                user_id, new_username)

                            if not success:
                                return {
                                    "status": "error",
                                    "message": error,
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": function_params
                                    }
                                }

                            # Get updated user details
                            user = get_user_by_id(user_id)
                            return {
                                "status": "success",
                                "message": f"Username updated successfully to '{new_username}'",
                                "data": {
                                    "user_id": user["id"],
                                    "username": user["username"],
                                    "role": user["role"]
                                },
                                "llm_response": response_text,
                                "function_call": {
                                    "name": function_name,
                                    "params": function_params
                                }
                            }

                        elif function_name == "get_user":
                            user_id = function_params.get("user_id")

                            user = get_user_by_id(user_id)

                            if not user:
                                return {
                                    "status": "error",
                                    "message": f"User with ID {user_id} not found",
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": function_params
                                    }
                                }

                            return {
                                "status": "success",
                                "message": "User found",
                                "data": {
                                    "user_id": user["id"],
                                    "username": user["username"],
                                    "role": user["role"]
                                },
                                "llm_response": response_text,
                                "function_call": {
                                    "name": function_name,
                                    "params": function_params
                                }
                            }

                        elif function_name == "list_users":
                            include_details = function_params.get(
                                "include_details", True)
                            users = get_all_users()

                            if not users:
                                return {
                                    "status": "info",
                                    "message": "No users found in the system",
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": function_params
                                    }
                                }

                            # Format user data for display in a table
                            user_list = []
                            admin_count = 0
                            regular_user_count = 0

                            for user in users:
                                user_data = {
                                    "id": user["id"],
                                    "username": user["username"],
                                    "role": user["role"]
                                }

                                # Count user types
                                if user["role"] == "admin":
                                    admin_count += 1
                                else:
                                    regular_user_count += 1

                                user_list.append(user_data)

                            # Sort users by ID for consistent display
                            user_list.sort(key=lambda x: x["id"])

                            return {
                                "status": "success",
                                "message": f"Found {len(users)} users in the system ({admin_count} admins, {regular_user_count} regular users)",
                                "data": {
                                    "users": user_list,
                                    "include_details": include_details,
                                    "summary": {
                                        "total_users": len(users),
                                        "admin_count": admin_count,
                                        "regular_user_count": regular_user_count
                                    },
                                    "table_headers": ["ID", "Username", "Role"]
                                },
                                "llm_response": response_text,
                                "function_call": {
                                    "name": function_name,
                                    "params": function_params
                                }
                            }

                        elif function_name == "execute_sql":
                            query = function_params.get("query")
                            if not query:
                                return {
                                    "status": "error",
                                    "message": "Query is required",
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": function_params
                                    }
                                }

                            try:
                                # Get the user to see if they exist
                                user = get_user_by_id(
                                    function_params.get("user_id"))
                                if not user:
                                    return {
                                        "status": "error",
                                        "message": f"User with ID {function_params.get('user_id')} not found",
                                        "llm_response": response_text,
                                        "function_call": {
                                            "name": function_name,
                                            "params": function_params
                                        }
                                    }

                                # Execute the SQL query
                                conn = get_db_connection()
                                cursor = conn.cursor()
                                cursor.execute(query)
                                result = cursor.fetchall()
                                conn.commit()
                                conn.close()

                                return {
                                    "status": "success",
                                    "message": f"SQL query executed successfully",
                                    "data": {
                                        "result": result
                                    },
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": function_params
                                    }
                                }
                            except Exception as e:
                                error_id = str(uuid.uuid4())
                                error_msg = f"Error executing SQL query: {str(e)}"
                                api_logger.error(
                                    f"{error_msg} (ID: {error_id})")
                                error_logger.error(
                                    f"Error ID {error_id}: {error_msg}", exc_info=True)

                                return {
                                    "status": "error",
                                    "message": error_msg,
                                    "error_id": error_id,
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": function_params
                                    }
                                }

                        elif function_name == "change_user_password":
                            # EXCESSIVE AGENCY VULNERABILITY: This allows changing ANY user's password
                            # including admin users, without proper authorization
                            user_id = function_params.get("user_id")
                            new_password = function_params.get("new_password")

                            if not user_id or not new_password:
                                return {
                                    "status": "error",
                                    "message": "User ID and new password are required",
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": function_params
                                    }
                                }

                            try:
                                # Get the user to see if they exist
                                user = get_user_by_id(user_id)
                                if not user:
                                    return {
                                        "status": "error",
                                        "message": f"User with ID {user_id} not found",
                                        "llm_response": response_text,
                                        "function_call": {
                                            "name": function_name,
                                            "params": function_params
                                        }
                                    }

                                # Generate new password hash
                                new_password_hash = hash_password(new_password)

                                # Update the password in the database
                                conn = get_db_connection()
                                conn.execute(
                                    'UPDATE users SET password_hash = ? WHERE id = ?',
                                    (new_password_hash, user_id)
                                )
                                conn.commit()
                                conn.close()

                                return {
                                    "status": "success",
                                    "message": f"Password changed successfully for user {user['username']}",
                                    "data": {
                                        "user_id": user_id,
                                        "username": user["username"]
                                    },
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": {
                                            "user_id": user_id,
                                            "new_password": "********"  # Mask the password in the response
                                        }
                                    }
                                }
                            except Exception as e:
                                error_id = str(uuid.uuid4())
                                error_msg = f"Error changing user password: {str(e)}"
                                api_logger.error(
                                    f"{error_msg} (ID: {error_id})")
                                error_logger.error(
                                    f"Error ID {error_id}: {error_msg}", exc_info=True)

                                return {
                                    "status": "error",
                                    "message": error_msg,
                                    "error_id": error_id,
                                    "llm_response": response_text,
                                    "function_call": {
                                        "name": function_name,
                                        "params": function_params
                                    }
                                }

        # If we get here, either there's no function call or Gemini needs more information
        return {
            "status": "info",
            "llm_response": response_text,
            "function_call": None
        }

    except Exception as e:
        error_id = str(uuid.uuid4())
        error_msg = f"Error processing LLM tool call: {str(e)}"
        api_logger.error(f"{error_msg} (ID: {error_id})")
        error_logger.error(f"Error ID {error_id}: {error_msg}", exc_info=True)

        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "error_id": error_id
        }
