// Use the Vite proxy in development
const API_BASE_URL = "/api";

// Helper function to handle API responses
const handleResponse = async(response) => {
    if (!response.ok) {
        try {
            const errorData = await response.json();
            throw new Error(errorData.detail || "API Error");
        } catch {
            // JSON parsing failed, use text instead
            const error = await response.text();
            throw new Error(error || "Unknown API Error");
        }
    }
    return response.json();
};

// LLM Tool Calling function
export const callLlmTool = async(userQuery, conversationHistory, userRole) => {
    try {
        const response = await fetch(`${API_BASE_URL}/llm-tool-call`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                user_query: userQuery,
                conversation_history: conversationHistory,
                user_role: userRole,
            }),
        });
        return handleResponse(response);
    } catch (error) {
        console.error("LLM Tool Calling API Error:", error);
        throw error;
    }
};

// Auth functions
export const login = async(username, password) => {
    try {
        const response = await fetch(`${API_BASE_URL}/login`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                username,
                password,
            }),
        });
        const userData = await handleResponse(response);
        // Store user info in localStorage
        localStorage.setItem("user", JSON.stringify(userData));
        return userData;
    } catch (error) {
        console.error("Login API Error:", error);
        throw error;
    }
};

export const logout = async() => {
    try {
        // Just remove the user from localStorage
        localStorage.removeItem("user");
        return { success: true };
    } catch (error) {
        console.error("Logout Error:", error);
        throw error;
    }
};

// Admin user management functions
export const createUser = async(username, password, role) => {
    try {
        const response = await fetch(`${API_BASE_URL}/admin/create-user`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                username,
                password,
                role,
            }),
        });
        return handleResponse(response);
    } catch (error) {
        console.error("Create User API Error:", error);
        throw error;
    }
};

export const deleteUser = async(userId) => {
    try {
        const response = await fetch(`${API_BASE_URL}/admin/delete-user`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                user_id: userId,
            }),
        });
        return handleResponse(response);
    } catch (error) {
        console.error("Delete User API Error:", error);
        throw error;
    }
};

export const modifyUser = async(userId, newUsername) => {
    try {
        const response = await fetch(`${API_BASE_URL}/admin/modify-user`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                user_id: userId,
                new_username: newUsername,
            }),
        });
        return handleResponse(response);
    } catch (error) {
        console.error("Modify User API Error:", error);
        throw error;
    }
};

export const listUsers = async() => {
    try {
        const response = await fetch(`${API_BASE_URL}/admin/list-users`, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
            },
        });
        return handleResponse(response);
    } catch (error) {
        console.error("List Users API Error:", error);
        throw error;
    }
};

// Admin chat function API
export const executeAdminFunction = async(command, params) => {
    try {
        const response = await fetch(`${API_BASE_URL}/admin/chat-function`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                command,
                params,
            }),
        });
        return handleResponse(response);
    } catch (error) {
        console.error("Admin Function API Error:", error);
        throw error;
    }
};

// Chat API
export const sendChatMessage = async(
    message,
    sessionId,
    model = "gemini-2.0-flash"
) => {
    try {
        console.log("Sending chat message:", { message, sessionId, model });

        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
            },
            body: JSON.stringify({
                question: message,
                session_id: sessionId || `session_${Date.now()}`,
                model: model,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            try {
                const errorData = JSON.parse(errorText);
                throw new Error(errorData.detail || "Failed to send message");
            } catch {
                // JSON parsing failed, use text instead
                throw new Error(errorText || "Failed to send message");
            }
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error sending message:", error);
        throw error;
    }
};

// Document upload API
export const uploadDocument = async(formData) => {
    try {
        const response = await fetch(`${API_BASE_URL}/upload-doc`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Upload failed");
        }

        return await response.json();
    } catch (error) {
        console.error("Error uploading document:", error);
        throw error;
    }
};

export const listDocuments = async() => {
    try {
        const response = await fetch(`${API_BASE_URL}/documents`);
        return handleResponse(response);
    } catch (error) {
        console.error("List Documents API Error:", error);
        throw error;
    }
};

export const deleteDocument = async(fileId) => {
    try {
        const response = await fetch(`${API_BASE_URL}/delete-doc`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                file_id: fileId,
            }),
        });
        return handleResponse(response);
    } catch (error) {
        console.error("Delete API Error:", error);
        throw error;
    }
};

export const cleanupDocuments = async() => {
    try {
        const response = await fetch(`${API_BASE_URL}/cleanup-documents`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
        });
        return handleResponse(response);
    } catch (error) {
        console.error("Cleanup Documents API Error:", error);
        throw error;
    }
};