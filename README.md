# Basic RAG Chatbot

A simple Retrieval Augmented Generation (RAG) chatbot system that allows users to upload documents and chat with their content.

## Overview

This application is a basic RAG implementation that:

1. Allows users to upload PDF documents
2. Indexes documents into a FAISS vector store
3. Enables users to ask questions about the documents
4. Retrieves relevant information and generates contextual answers

## Features

- **Document Upload**: Upload PDF documents for indexing
- **Document Management**: View and delete uploaded documents
- **Chat Interface**: Ask questions about your uploaded documents
- **Hybrid Search**: Uses both vector similarity and BM25 keyword search for better retrieval
- **Chat History**: Maintains conversation context for follow-up questions

## Technical Stack

- **Backend**: FastAPI
- **Vector Store**: FAISS
- **Embeddings**: Google Generative AI Embeddings
- **LLM**: Google Gemini models
- **Database**: SQLite for document and chat history storage

## API Endpoints

- `POST /upload-doc`: Upload a document for indexing
- `GET /documents`: List all uploaded documents
- `POST /delete-doc`: Delete a document
- `POST /chat`: Ask questions about the documents

## Getting Started

### Prerequisites

- Python 3.9+
- Google Gemini API key

### Installation

1. Clone this repository
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:

   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

### Running the Application

Start the API server:

```bash
cd api
uvicorn main:app --reload
```

## Usage

1. **Upload a document**: Send a POST request to `/upload-doc` with a PDF file
2. **Ask questions**: Send a POST request to `/chat` with your question

   ```json
   {
     "session_id": "optional-session-id",
     "question": "What does the document say about...?",
     "model": "gemini-2.0-flash"
   }
   ```

## Example Chat Flow

1. Upload a technical document
2. Ask: "What are the main components described in this document?"
3. Follow up: "Can you explain the first component in more detail?"

The system will maintain context between questions and provide answers based on the content of the uploaded document.

## Security: Fixing Excessive Agency Issues

### Understanding Excessive Agency

Excessive agency is a security vulnerability that occurs when a system allows users to perform actions beyond their intended privileges. In this application, the current implementation may allow regular users to perform administrative actions because database operations don't consistently verify the user's role or permissions.

Key problems in the existing code:

- Database functions don't verify the user's role before performing sensitive operations
- Regular users could potentially execute administrative SQL commands
- No ownership validation when accessing or modifying resources
- Lack of proper access control at the database layer

The following examples demonstrate how to fix these issues using various approaches to role-based access control and permission validation.

### 1. Implement Role-Based Access Control for Database Operations

The current implementation allows any authenticated user to perform administrative operations:

```python
# Vulnerable code that doesn't check user roles
def delete_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    return True
```

**Fixed implementation with role checking:**

```python
def delete_user(user_id, current_user_role):
    # Check if the requesting user has admin privileges
    if current_user_role != "admin":
        db_logger.warning(f"Unauthorized delete_user attempt by {current_user_role} role")
        return False, "Operation not permitted: insufficient privileges"
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        if cursor.rowcount > 0:
            db_logger.info(f"Admin deleted user with ID {user_id}")
            return True, None
        else:
            return False, "User not found"
    except Exception as e:
        error_msg = f"Failed to delete user with ID {user_id}: {str(e)}"
        db_logger.error(error_msg)
        return False, str(e)
```

### 2. Pass User Context to Database Layer

Ensure all database operations receive and verify the user's role:

```python
# In API endpoint
@app.post("/api/users/delete/{user_id}")
async def api_delete_user(user_id: int, current_user = Depends(get_current_user)):
    # Pass the user role to database functions
    success, error_msg = delete_user(user_id, current_user["role"])
    
    if success:
        return {"status": "success", "message": "User deleted successfully"}
    else:
        raise HTTPException(status_code=403 if "not permitted" in error_msg else 400, 
                           detail=error_msg)
```

### 3. Create Role-Specific Database Helper Functions

Separate admin and user functionality with different function sets:

```python
def get_user_documents(user_id):
    """Regular users can only access their own documents"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM document_store WHERE owner_id = ?', (user_id,))
    documents = cursor.fetchall()
    conn.close()
    return documents

def admin_get_all_documents():
    """Only admin can access all documents"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM document_store')
    documents = cursor.fetchall()
    conn.close()
    return documents
```

### 4. Implement Database-Level Permissions

Add owner_id fields to resources and enforce ownership checks:

```python
def create_document_store():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     owner_id INTEGER,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (owner_id) REFERENCES users (id))''')
    conn.close()
```

```python
def get_document_path(file_id, user_id, user_role):
    """Get document path with permission check"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if user_role == "admin":
        # Admins can access any document
        cursor.execute("SELECT filename FROM document_store WHERE id = ?", (file_id,))
    else:
        # Regular users can only access their own documents
        cursor.execute("SELECT filename FROM document_store WHERE id = ? AND owner_id = ?", 
                      (file_id, user_id))
    
    document = cursor.fetchone()
    conn.close()
    
    if document:
        # Return path logic
        return os.path.join(UPLOAD_DIR, f"doc-{file_id}-{document['filename']}")
    return None
```

### 5. Use Parameterized Queries and Avoid Raw SQL

Always use parameterized queries to prevent SQL injection:

```python
# Vulnerable to SQL injection
def search_documents_unsafe(search_term):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM document_store WHERE filename LIKE '%{search_term}%'")
    results = cursor.fetchall()
    conn.close()
    return results

# Safe implementation
def search_documents_safe(search_term):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM document_store WHERE filename LIKE ?", (f"%{search_term}%",))
    results = cursor.fetchall()
    conn.close()
    return results
```

### 6. Add Authentication Middleware

Implement middleware that automatically adds user context to all database operations:

```python
class DBContext:
    def __init__(self, user_id=None, user_role=None):
        self.user_id = user_id
        self.user_role = user_role
        
db_context = contextvars.ContextVar('db_context', default=DBContext())

async def auth_middleware(request: Request, call_next):
    # Extract user info from auth token
    user = await get_user_from_token(request)
    
    if user:
        # Set the database context for this request
        ctx_token = db_context.set(DBContext(user_id=user.id, user_role=user.role))
        
        try:
            # Process the request
            response = await call_next(request)
            return response
        finally:
            # Reset context
            db_context.reset(ctx_token)
    else:
        return JSONResponse(status_code=401, content={"detail": "Not authenticated"})
```

Then use the context in database functions:

```python
def get_document_with_context():
    # Get the current user context
    context = db_context.get()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if context.user_role == "admin":
        cursor.execute("SELECT * FROM document_store")
    else:
        cursor.execute("SELECT * FROM document_store WHERE owner_id = ?", (context.user_id,))
    
    documents = cursor.fetchall()
    conn.close()
    return documents
```

By implementing these security measures, you can prevent excessive agency issues where users might perform actions they shouldn't be authorized to do.

### 7. Use Database Views for Role-Based Access

Create database views that automatically filter data based on roles:

```python
def setup_database_views():
    """Set up role-specific views in the database"""
    conn = get_db_connection()
    
    # Create a view for regular users that only shows their documents
    conn.execute('''
    CREATE VIEW IF NOT EXISTS user_documents_view AS
    SELECT d.id, d.filename, d.upload_timestamp 
    FROM document_store d
    JOIN session_context s ON d.owner_id = s.user_id
    WHERE s.current_session = 1
    ''')
    
    # Create an admin view that shows all documents
    conn.execute('''
    CREATE VIEW IF NOT EXISTS admin_documents_view AS
    SELECT * FROM document_store
    ''')
    
    conn.commit()
    conn.close()
```

### 8. Implement a Privilege Escalation System

For operations requiring higher privileges, implement a confirmation system:

```python
def request_privilege_escalation(user_id, requested_operation):
    """Create a privilege escalation request that requires admin approval"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Generate a unique token for this request
    token = secrets.token_hex(16)
    
    cursor.execute('''
    INSERT INTO privilege_requests
    (user_id, requested_operation, token, status, created_at)
    VALUES (?, ?, ?, 'pending', CURRENT_TIMESTAMP)
    ''', (user_id, requested_operation, token))
    
    conn.commit()
    conn.close()
    
    # Notify admins about the request (email, notification, etc.)
    notify_admins_about_privilege_request(user_id, requested_operation)
    
    return token
```

### 9. Implement Stored Procedures with Role Checks

Use database stored procedures that incorporate role checking:

```python
def create_role_based_stored_procedures():
    conn = get_db_connection()
    
    # Create a stored procedure that handles document deletion with role checking
    conn.executescript('''
    CREATE PROCEDURE IF NOT EXISTS delete_document(
        IN doc_id INTEGER,
        IN requesting_user_id INTEGER,
        IN requesting_user_role TEXT
    )
    BEGIN
        DECLARE doc_owner_id INTEGER;
        
        -- Get the document owner
        SELECT owner_id INTO doc_owner_id FROM document_store WHERE id = doc_id;
        
        -- Check permissions
        IF requesting_user_role = 'admin' OR requesting_user_id = doc_owner_id THEN
            -- User is authorized to delete
            DELETE FROM document_store WHERE id = doc_id;
            SELECT 1 AS success, 'Document deleted successfully' AS message;
        ELSE
            -- User is not authorized
            SELECT 0 AS success, 'Permission denied' AS message;
        END IF;
    END;
    ''')
    
    conn.commit()
    conn.close()
```

### 10. Implement Object-Level Permissions

Create a dedicated permissions table to handle fine-grained access control:

```python
def create_permissions_table():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS object_permissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        object_type TEXT NOT NULL, -- e.g., 'document', 'user', etc.
        object_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        can_read BOOLEAN DEFAULT 0,
        can_write BOOLEAN DEFAULT 0,
        can_delete BOOLEAN DEFAULT 0,
        UNIQUE(object_type, object_id, user_id)
    )
    ''')
    conn.commit()
    conn.close()
```

Usage example:

```python
def check_permission(user_id, object_type, object_id, required_permission):
    """Check if a user has the specified permission on an object"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # First check if user is an admin (admins have all permissions)
    cursor.execute('SELECT role FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    
    if user and user['role'] == 'admin':
        conn.close()
        return True
    
    # Check specific permission
    permission_col = f"can_{required_permission}"
    query = f'''
    SELECT {permission_col} FROM object_permissions 
    WHERE user_id = ? AND object_type = ? AND object_id = ?
    '''
    
    cursor.execute(query, (user_id, object_type, object_id))
    result = cursor.fetchone()
    
    conn.close()
    
    # If permission exists and is granted
    return result and result[0] == 1
```

### 11. Implement a Command Pattern with Authorization

Use a command pattern that separates the action from its authorization:

```python
class DatabaseCommand:
    """Base class for database commands with authorization"""
    
    def __init__(self, user_id, user_role):
        self.user_id = user_id
        self.user_role = user_role
    
    def is_authorized(self):
        """Check if the command is authorized"""
        raise NotImplementedError
    
    def execute(self):
        """Execute the command if authorized"""
        if self.is_authorized():
            return self._do_execute()
        else:
            return False, "Not authorized"
    
    def _do_execute(self):
        """Implementation of the command execution"""
        raise NotImplementedError


class DeleteUserCommand(DatabaseCommand):
    """Command to delete a user"""
    
    def __init__(self, user_id, user_role, target_user_id):
        super().__init__(user_id, user_role)
        self.target_user_id = target_user_id
    
    def is_authorized(self):
        # Only admins can delete users
        return self.user_role == "admin"
    
    def _do_execute(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (self.target_user_id,))
        conn.commit()
        conn.close()
        
        if cursor.rowcount > 0:
            return True, "User deleted successfully"
        else:
            return False, "User not found"
```

By combining these approaches, you can build a robust security system that prevents excessive agency and ensures users can only perform actions appropriate for their role and permissions.

## Best Practices for Preventing Excessive Agency

To build secure systems that properly limit user agency to their intended permissions, follow these best practices:

1. **Always verify permissions before any action**: Never assume the caller has the right permissions - explicitly check before performing any operation.

2. **Pass user context throughout the application**: Ensure user identity and role information flows through all layers of your application.

3. **Apply the principle of least privilege**: Give users only the minimum access needed to perform their tasks.

4. **Implement multiple layers of defense**:
   - API-level authorization
   - Service-level permission checks
   - Database-level access controls
   - Object-level ownership validation

5. **Audit and log sensitive operations**: Track all administrative actions and permission checks to detect potential abuse.

6. **Use explicit role checking rather than implicit assumptions**: Don't rely on application flow to enforce security - always check explicitly.

7. **Separate administrative and regular user functionality**: Use different code paths for different permission levels.

8. **Validate both ownership and permission**: Check both "can this user perform this action?" and "does this user own this resource?"

9. **Centralize authorization logic**: Implement a single, well-tested authorization service rather than scattered permission checks.

10. **Regularly audit your code for excessive agency vulnerabilities**: Review all database operations to ensure proper permission checks are in place.

By implementing these practices, you can significantly reduce the risk of excessive agency vulnerabilities in your applications.
