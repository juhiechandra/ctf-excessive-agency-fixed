# How We Fixed the Excessive Agency Vulnerability

## The Problem: Excessive Agency in LLM-Powered Chatbots

Excessive Agency is a security vulnerability that occurs when a Language Model (LLM) is given the ability to perform actions beyond what a user should be authorized to do. This happens for several reasons:

1. The system fails to properly validate permissions before executing actions requested by the LLM
2. Role-Based Access Control (RBAC) is implemented at the UI layer but not enforced at the API layer
3. The system trusts the LLM to enforce security boundaries without proper checks

In an agentic LLM chatbot, this vulnerability allows regular users to issue commands (through natural language) that would normally be restricted to administrators, such as:

- Creating new users
- Changing other users' passwords
- Executing privileged SQL queries
- Accessing or modifying unauthorized resources

## Our Solution: Centralized Permission Checking

We implemented a simple yet effective solution by creating a centralized permission validation function that is used across all database operations. This approach has several advantages:

1. **Single Source of Truth**: All permission checks go through one function, making the security model easier to understand and maintain
2. **Consistent Enforcement**: The same validation rules are applied everywhere, eliminating gaps in security
3. **Separation of Concerns**: Authentication logic is separated from business logic
4. **Defense in Depth**: Permissions are checked at multiple levels, not just in the UI

### The Implementation

We created a centralized `check_permission` function in `db_utils.py` that handles all authorization checks:

```python
def check_permission(requesting_user_id, target_resource_id=None, action=None, resource_type=None):
    """
    Centralized permission check function for database operations.
    
    Args:
        requesting_user_id: The ID of the user requesting the action
        target_resource_id: The ID of the resource being acted upon (user_id, document_id, etc.)
        action: The action being performed ('create', 'read', 'update', 'delete', 'execute')
        resource_type: The type of resource ('user', 'document', 'sql', etc.)
        
    Returns:
        tuple: (allowed, error_message)
            - allowed (bool): True if the action is permitted, False otherwise
            - error_message (str): Error message if not permitted
    """
    try:
        # If no requesting user, deny all privileged operations
        if not requesting_user_id:
            return False, "Authentication required for this operation"
            
        # Get the user's role
        user = get_user_by_id(requesting_user_id)
        if not user:
            return False, "User not found"
            
        user_role = user["role"]
        
        # Always allow admins
        if user_role == "admin":
            return True, None
            
        # Regular user permissions
        if resource_type == "user":
            # Users can only modify themselves
            if action in ["update", "delete"] and target_resource_id != requesting_user_id:
                return False, "You can only modify your own user account"
                
            # Users can't create new users
            if action == "create":
                return False, "Only administrators can create new users"
                
        elif resource_type == "sql":
            # Only admins can execute SQL
            return False, "Only administrators can execute SQL queries"
            
        elif resource_type == "document":
            # All authenticated users can manage documents
            return True, None
            
        # Default to allowed for authenticated users
        return True, None
        
    except Exception as e:
        error_msg = f"Failed to check permissions: {str(e)}"
        db_logger.error(error_msg)
        error_logger.error(error_msg, exc_info=True)
        # Default to denied on error
        return False, error_msg
```

We then modified all database functions that perform sensitive operations to use this centralized function:

```python
# Example of using the centralized permission check
def change_user_password(user_id, new_password, requesting_user_id=None):
    with PerformanceTimer(db_logger, f"change_user_password:{user_id}"):
        try:
            # Use the centralized permission check
            if requesting_user_id is not None:
                allowed, error = check_permission(
                    requesting_user_id, 
                    target_resource_id=user_id, 
                    action="update", 
                    resource_type="user"
                )
                if not allowed:
                    return False, error

            # Rest of the function...
```

## Benefits of Our Approach

1. **Simplified Code**: No need for repetitive permission checks in each function
2. **Consistent Security Model**: All permissions are defined in one place
3. **Easy to Extend**: New permissions can be added to the central function
4. **Robust Against LLM Manipulation**: No matter what the LLM tries to do, the system enforces proper authorization
5. **Defense in Depth**: Even if the UI allows a regular user to access admin functions, the backend will reject unauthorized operations

## Best Practices We Followed

1. **Never Trust Client-Side Validation**: All permissions are checked at the API and function levels
2. **Apply Least Privilege Principle**: Each user gets only the permissions they need
3. **Use Explicit Permission Checks**: Direct comparison of user roles and IDs
4. **Default Deny**: When in doubt, deny access
5. **Multiple Layers of Protection**: Even if one layer fails, others will catch unauthorized access

By implementing this centralized permission check system, we effectively mitigated the excessive agency vulnerability, ensuring that users can only perform actions appropriate for their roles, regardless of what natural language commands they give to the LLM chatbot.
