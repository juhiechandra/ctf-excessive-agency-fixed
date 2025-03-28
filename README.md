# Secure RAG Chatbot: Fixed Excessive Agency Vulnerability

This repository contains a Retrieval Augmented Generation (RAG) chatbot system designed to demonstrate and fix the "Excessive Agency" vulnerability in AI-powered chatbots.

## What is Excessive Agency?

Excessive Agency is a security vulnerability that occurs when a Language Model (LLM) is given the ability to perform actions beyond what a user should be authorized to do. This commonly happens when:

1. The system fails to properly validate permissions before executing actions requested by the LLM
2. The system trusts the LLM to enforce security boundaries without proper checks
3. Role-Based Access Control (RBAC) is implemented at the UI layer but not enforced at the API layer

In this demo application, we demonstrate how the vulnerability works and how it can be properly fixed.

## Key Components

The application consists of:

- **Backend API (FastAPI)**: Provides endpoints for document management, user management, and chat functionality
- **Frontend (React)**: Provides a user interface for interacting with the chatbot
- **RAG System**: Uses LangChain with FAISS vector database for document retrieval and generation
- **User Management**: Simple role-based system with 'user' and 'admin' roles

## How the Vulnerability Was Fixed

The following key fixes were implemented to address the excessive agency vulnerability:

1. **Centralized Permission Checking System**:
   - Implemented a `check_permission` function that serves as a single source of truth for all authorization checks
   - All database operations now funnel through this centralized permission validation function
   - Consistent enforcement of security rules across the entire application

2. **Proper Role-Based Access Control (RBAC) at API Layer**:
   - Each API endpoint now validates the user's role before performing sensitive operations
   - Regular users can only perform actions appropriate for their role
   - Admin-only functions are properly secured with explicit role checks

3. **Function-Call Guards**:
   - Added explicit role checks before executing specific functions
   - For example, in the `execute_sql` function, added: `if not requesting_user_id or user_role != "admin"`
   - For password changes: `if user_role != "admin" and (not requesting_user_id or requesting_user_id != target_user_id)`

4. **Clear System Prompts with Role Boundaries**:
   - The system prompt now clearly defines what functions each role can access
   - Explicit instructions for what regular users can and cannot do
   - Proper handling of sensitive operations like password changes

5. **Permission Validation at Multiple Levels (Defense in Depth)**:
   - User authentication and role verification at API endpoints
   - Additional validation at the function execution level
   - Security checks independent of LLM behavior
   - Default deny policy for unauthorized operations

## Running the Application

### Prerequisites

- Python 3.9+
- Node.js 16+
- [Google API Key](https://makersuite.google.com/) for Gemini model access

### Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/ctf-excessive-agency-fixed.git
cd ctf-excessive-agency-fixed
```

2. Set up the backend:

```bash
cd api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the api directory with:

```
GOOGLE_API_KEY=your_api_key_here
```

4. Set up the frontend:

```bash
cd ../frontend
npm install
```

### Running

1. Start the backend:

```bash
cd api
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn main:app --reload
```

2. Start the frontend:

```bash
cd frontend
npm run dev
```

3. Open <http://localhost:5173> in your browser

## Best Practices for Preventing Excessive Agency

1. **Never Trust the LLM to Enforce Security**: Always validate permissions at the API and function levels.
2. **Implement RBAC at API Endpoints**: Each endpoint should validate the user's role.
3. **Use Explicit Permission Checks**: Directly compare user roles and IDs before performing sensitive operations.
4. **Apply Least Privilege Principle**: Give users access only to what they need.
5. **Use Admin-Only Functions**: Create separate functions for administrative tasks.
6. **Validate Parameters**: Check that users can only modify their own data (unless they're admins).
7. **Sanitize Inputs**: Never directly execute SQL or other commands from LLM output without validation.
8. **Log and Monitor**: Keep detailed logs of all actions for audit purposes.
9. **Regular Security Testing**: Test for privilege escalation regularly.
10. **Default Deny Policy**: When in doubt about permissions, deny access rather than allowing it.

## License

MIT

## Contributors

This is a demonstration project created for educational purposes.
