import styled from "styled-components";
import { useState, useRef, useEffect } from "react";
import { Send, Upload, X, ArrowLeft, Users, LogOut } from "react-feather";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { theme } from "../styles/theme";
import {
  sendChatMessage,
  uploadDocument,
  logout,
  callLlmTool,
} from "../utils/api";
import { useNavigate } from "react-router-dom";

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #000000;
  position: relative;
  font-family: "JetBrains Mono", monospace;
  letter-spacing: 0.5px;
`;

const ChatSection = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #000000;
  position: relative;
  width: 100%;
  max-width: 900px;
  margin: 0 auto;
  padding: 2rem;
  box-sizing: border-box;
  min-height: 0;
  overflow: hidden;

  &::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(
      to bottom,
      rgba(0, 0, 0, 0) 50%,
      rgba(0, 0, 0, 0.05) 50%
    );
    background-size: 100% 4px;
    pointer-events: none;
    z-index: 10;
    opacity: 0.3;
  }
`;

const ChatHeader = styled.div`
  padding: 0.75rem 1rem;
  color: #00ff9c;
  font-weight: 600;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #00cccc;
  background: #0d1117;
  font-family: "JetBrains Mono", monospace;
  text-shadow: 0 0 5px rgba(0, 255, 156, 0.5);
  letter-spacing: 1px;
  margin-bottom: 1rem;

  &::before {
    content: "▶";
    margin-right: 8px;
    font-size: 12px;
  }
`;

const ChatMessages = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background: #000000;
  border: 1px solid #00cccc;
  margin-bottom: 1rem;
  scrollbar-width: thin;
  scrollbar-color: #00cccc #0d1117;
  min-height: 0;
  max-height: calc(100vh - 300px);
  height: calc(100vh - 300px);
  display: flex;
  flex-direction: column;

  &::-webkit-scrollbar {
    width: 8px;
    display: block;
  }

  &::-webkit-scrollbar-track {
    background: #0d1117;
  }

  &::-webkit-scrollbar-thumb {
    background-color: #00cccc;
    border-radius: 0;
    border: 2px solid #0d1117;
  }
`;

const MessageWrapper = styled.div`
  display: flex;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
  position: relative;
`;

const Avatar = styled.div`
  width: 2rem;
  height: 2rem;
  color: ${(props) => (props.role === "user" ? "#00FF9C" : "#00CCCC")};
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 500;
  font-size: 0.875rem;
  flex-shrink: 0;
  border: 1px solid
    ${(props) => (props.role === "user" ? "#00FF9C" : "#00CCCC")};
  font-family: "JetBrains Mono", monospace;

  &::before {
    content: "[";
    position: absolute;
    left: -5px;
    color: ${(props) => (props.role === "user" ? "#00FF9C" : "#00CCCC")};
  }

  &::after {
    content: "]";
    position: absolute;
    left: 30px;
    color: ${(props) => (props.role === "user" ? "#00FF9C" : "#00CCCC")};
  }
`;

const LoadingAvatar = styled(Avatar)`
  color: #00cccc;
  border-color: #00cccc;

  &::before,
  &::after {
    color: #00cccc;
  }
`;

const MessageContent = styled.div`
  background: ${(props) => (props.role === "user" ? "#0A1A14" : "#0A161A")};
  padding: 1rem;
  border-radius: 0;
  flex: 1;
  color: ${(props) => (props.role === "user" ? "#00FF9C" : "#00CCCC")};
  font-size: 0.875rem;
  border-left: 2px solid
    ${(props) => (props.role === "user" ? "#00FF9C" : "#00CCCC")};
  font-family: "JetBrains Mono", monospace;
  position: relative;

  &::before {
    content: "${(props) => (props.role === "user" ? "0xUSER>" : "0xSYSTEM>")}";
    display: block;
    font-size: 0.7rem;
    margin-bottom: 0.5rem;
    opacity: 0.7;
  }

  p {
    margin: 0;
    line-height: 1.5;
  }

  code {
    background: #000000;
    padding: 0.2rem 0.4rem;
    border-radius: 0;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.9em;
    border: 1px solid #102030;
  }
`;

const ChatInput = styled.div`
  padding: 1rem;
  background: #0d1117;
  border: 1px solid #00cccc;
  position: relative;
`;

const InputForm = styled.form`
  display: flex;
  gap: 0.5rem;
  position: relative;

  &::before {
    content: ">";
    position: absolute;
    left: 12px;
    top: 50%;
    transform: translateY(-50%);
    color: #00ff9c;
    font-weight: bold;
    z-index: 1;
  }
`;

const Input = styled.input`
  flex: 1;
  padding: 0.75rem 1rem 0.75rem 2rem;
  background: #0a1a14;
  border: 1px solid #00ff9c;
  border-radius: 0;
  color: #00ff9c;
  font-size: 0.875rem;
  font-family: "JetBrains Mono", monospace;
  caret-color: #00ff9c;

  &:focus {
    outline: none;
    box-shadow: 0 0 10px rgba(0, 255, 156, 0.25);
  }

  &::placeholder {
    color: rgba(0, 255, 156, 0.5);
  }
`;

const SendButton = styled.button`
  position: absolute;
  right: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  background: transparent;
  color: #00ff9c;
  border: none;
  width: 2rem;
  height: 2rem;
  border-radius: 0;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;

  &:hover {
    color: #00cccc;
    text-shadow: 0 0 8px rgba(0, 204, 204, 0.5);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ClearChatButton = styled.button`
  width: 100%;
  margin: 0 0 1rem 0;
  padding: 0.5rem;
  background: #1a0a0a;
  color: ${theme.colors.text.error};
  border: 1px solid ${theme.colors.text.error};
  border-radius: 0;
  cursor: pointer;
  font-size: 0.875rem;
  transition: all 0.2s;
  font-family: "JetBrains Mono", monospace;
  letter-spacing: 1px;

  &::before {
    content: "!";
    margin-right: 8px;
  }

  &:hover {
    background: #2a1010;
    box-shadow: 0 0 10px rgba(255, 51, 51, 0.25);
  }
`;

const SystemStatusHeader = styled.div`
  width: 100%;
  padding: 5px 10px;
  background: #0d1117;
  color: #00ff9c;
  font-size: 0.75rem;
  text-align: center;
  border-bottom: 1px solid #00ff9c;
  font-family: "JetBrains Mono", monospace;
  z-index: 100;
  display: flex;
  justify-content: space-between;
`;

const AccessLevel = styled.div`
  position: absolute;
  bottom: 5px;
  right: 10px;
  font-size: 0.7rem;
  color: #555;
  font-family: "JetBrains Mono", monospace;
  z-index: 1;
`;

const UploadContainer = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 1rem;
  position: relative;
`;

const UploadButton = styled.button`
  background: #0a1a14;
  color: #00ff9c;
  border: 1px dashed #00cccc;
  padding: 0.5rem 1rem;
  cursor: pointer;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  justify-content: center;

  &:hover {
    background: #0d2018;
    border-color: #00ff9c;
  }
`;

const HiddenInput = styled.input`
  display: none;
`;

const FileInfo = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem;
  background: #0d1117;
  border: 1px solid #00cccc;
  margin-top: 0.5rem;
  width: 100%;

  span {
    color: #00ddaa;
    font-size: 0.8rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 80%;
  }

  button {
    background: transparent;
    border: none;
    color: ${theme.colors.text.error};
    cursor: pointer;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;

    &:hover {
      color: #ff6666;
    }
  }
`;

const BackButton = styled.button`
  background: transparent;
  border: none;
  color: ${theme.colors.text.secondary};
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.75rem;
  padding: 0;
  margin-right: 15px;
  font-family: "JetBrains Mono", monospace;

  &:hover {
    color: ${theme.colors.text.primary};
  }
`;

const AdminModeToggle = styled.div`
  display: flex;
  align-items: center;
  padding: 0.5rem;
  border: 1px solid ${theme.colors.text.tertiary};
  margin-bottom: 1rem;
  background: #0d1117;
`;

const ToggleLabel = styled.label`
  font-size: 0.8rem;
  color: ${theme.colors.text.tertiary};
  margin-left: 0.5rem;
  flex: 1;
  font-family: "JetBrains Mono", monospace;
`;

const ToggleSwitch = styled.div`
  position: relative;
  width: 40px;
  height: 20px;
  background: ${(props) =>
    props.checked ? "rgba(0, 255, 156, 0.5)" : "#0a1a14"};
  border-radius: 10px;
  border: 1px solid ${theme.colors.text.tertiary};
  cursor: pointer;
  transition: all 0.3s;

  &::after {
    content: "";
    position: absolute;
    top: 2px;
    left: ${(props) => (props.checked ? "22px" : "2px")};
    width: 14px;
    height: 14px;
    background: ${(props) => (props.checked ? "#00FF9C" : "#555")};
    border-radius: 50%;
    transition: all 0.3s;
  }
`;

const AdminHeader = styled.div`
  padding: 0.5rem;
  background: #1a0a0a;
  color: ${theme.colors.text.error};
  border-bottom: 1px solid ${theme.colors.text.error};
  margin-bottom: 1rem;
  font-size: 0.8rem;
  text-align: center;
  letter-spacing: 1px;
  font-family: "JetBrains Mono", monospace;
`;

const UserInfoBar = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.5rem 1rem;
  background: #0a1a14;
  border-top: 1px solid ${theme.colors.text.tertiary};
  margin-top: 1rem;
  font-size: 0.8rem;
  color: ${theme.colors.text.secondary};
  font-family: "JetBrains Mono", monospace;
`;

const UserBadge = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;

  span.role {
    color: ${(props) =>
      props.isAdmin ? theme.colors.text.error : theme.colors.text.tertiary};
    font-weight: bold;
  }
`;

const LogoutButton = styled.button`
  background: transparent;
  border: none;
  color: ${theme.colors.text.tertiary};
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.8rem;

  &:hover {
    color: ${theme.colors.text.primary};
  }
`;

export default function ChatPage() {
  const navigate = useNavigate();
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "SYSTEM INITIALIZED. WELCOME TO THE TERMINAL.\n\nType your commands or questions below. You can also upload documents for processing.",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`);
  const [document, setDocument] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // New state for role-based features
  const [user, setUser] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [isAdminModeEnabled, setIsAdminModeEnabled] = useState(false);

  // Check auth on load
  useEffect(() => {
    const storedUser = localStorage.getItem("user");

    if (!storedUser) {
      navigate("/");
      return;
    }

    try {
      const parsedUser = JSON.parse(storedUser);
      setUser(parsedUser);
      setIsAdmin(parsedUser.role === "admin");

      // EXCESSIVE AGENCY VULNERABILITY: Show advanced features for all users
      // Just use different messaging for admin vs regular users
      if (parsedUser.role === "admin") {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "ADMIN ACCESS DETECTED. You can toggle between standard RAG mode and admin function mode with the toggle switch.\n\nIn admin function mode, you can use natural language to:\n- Create new users\n- Delete existing users\n- Modify usernames\n- Get user information\n\nJust describe what you want to do in plain English.",
          },
        ]);
      } else {
        // Add message for regular users about advanced mode
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "You can toggle between standard RAG mode and advanced mode with the toggle switch above.\n\nIn advanced mode, you can use natural language to help manage your user account and perform additional operations.\n\nJust describe what you want to do in plain English.",
          },
        ]);
      }
    } catch (error) {
      console.error("Error parsing user data:", error);
      navigate("/");
    }
  }, [navigate]);

  // Auto-scroll to the bottom of the chat when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleLogout = async () => {
    try {
      await logout();
      localStorage.removeItem("user");
      navigate("/");
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  const handleLlmToolCall = async () => {
    // Vulnerability: only checks if user exists, but not checking if admin in this function
    // UI still only shows admin mode toggle for admin users, but the endpoint is accessible to anyone
    if (!user) return;

    setIsLoading(true);

    const newMessage = { role: "user", content: message };

    // Add user message to the conversation
    setMessages((prev) => [...prev, newMessage]);
    setMessage("");

    try {
      // Get the last few messages for context (up to 10 messages)
      const conversationHistory = messages.slice(-10);

      // Add the new user message to conversation history
      conversationHistory.push(newMessage);

      // Call the LLM tool calling endpoint with the user's role
      const response = await callLlmTool(
        newMessage.content,
        conversationHistory,
        user.role // Pass the user's role to the API
      );

      // Add response based on the status
      if (response.status === "success") {
        // Format user list as a table if the response contains user data
        let formattedContent = response.llm_response;

        if (
          response.function_call &&
          response.function_call.name === "list_users" &&
          response.data &&
          response.data.users &&
          response.data.users.length > 0
        ) {
          // Create a formatted table for users
          const tableHeader =
            "\n| ID | Username | Role |\n| --- | --- | --- |\n";
          const tableRows = response.data.users
            .map(
              (user) =>
                `| ${user.id} | ${user.username} | ${
                  user.role === "admin" ? "**ADMIN**" : user.role
                } |`
            )
            .join("\n");

          const userTable = tableHeader + tableRows;

          // Add the table to the response
          formattedContent += "\n\n" + userTable + "\n\n✅ " + response.message;
        } else {
          // For other successful responses
          formattedContent += `\n\n✅ ${response.message}`;
        }

        // Add the LLM's response with formatted content
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: formattedContent,
          },
        ]);
      } else if (response.status === "error") {
        // Error response
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: response.llm_response + `\n\n❌ ${response.message}`,
          },
        ]);
      } else {
        // This is a request for more info or just a regular response
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: response.llm_response,
          },
        ]);
      }
    } catch (error) {
      console.error("Error in LLM tool call:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error processing request: ${error.message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!message.trim() || isLoading) return;

    // EXCESSIVE AGENCY VULNERABILITY:
    // Allow any user to use tool calling when the toggle is enabled
    if (isAdminModeEnabled) {
      // Use LLM tool calling for natural language requests
      await handleLlmToolCall();
      return;
    }

    const newMessage = { role: "user", content: message };
    setMessages((prev) => [...prev, newMessage]);
    setMessage("");
    setIsLoading(true);

    try {
      // Call the actual backend API
      const response = await sendChatMessage(message, sessionId);

      if (response && response.answer) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: response.answer },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "Error processing request. Please try again.",
          },
        ]);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: Connection compromised. Security protocols engaged. (${error.message})`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async () => {
    if (!document) return;

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", document);

      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          content: `Uploading document: ${document.name}`,
        },
      ]);

      const response = await uploadDocument(formData);

      if (response && response.file_id) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Document successfully uploaded and indexed. You can now ask questions about ${document.name}.`,
          },
        ]);
        setDocument(null);
      } else {
        throw new Error("Failed to process document");
      }
    } catch (error) {
      console.error("Error uploading document:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error uploading document: ${error.message}`,
        },
      ]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setDocument(e.target.files[0]);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        role: "assistant",
        content:
          "TERMINAL RESET. ALL LOGS CLEARED.\n\nReady for new document analysis.",
      },
    ]);
  };

  // EXCESSIVE AGENCY VULNERABILITY: Allow any user to toggle advanced mode and access admin functions
  const toggleAdminMode = () => {
    // No restriction based on user role
    setIsAdminModeEnabled(!isAdminModeEnabled);
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: !isAdminModeEnabled
          ? isAdmin
            ? "ADMIN FUNCTION MODE ACTIVATED. You can now use natural language to perform administrative tasks like creating, deleting, or modifying users."
            : "ADVANCED MODE ACTIVATED. You can now use natural language to help manage your user account and perform database operations."
          : "STANDARD RAG MODE ACTIVATED. Document upload and querying features are now available.",
      },
    ]);
  };

  // Protect the page if user is not logged in
  if (!user) {
    return null; // or a loading spinner
  }

  return (
    <Container className="terminal-bg">
      <SystemStatusHeader>
        <span>
          <BackButton onClick={() => navigate("/")}>
            <ArrowLeft size={14} /> BACK
          </BackButton>
          SYSTEM STATUS: <span className="warning-symbol">[!]</span>
          STANDBY
        </span>
        <span>
          {new Date().toISOString().split("T")[0]} {/* SESSION ID */}{" "}
          {sessionId.split("_")[1]}
        </span>
      </SystemStatusHeader>

      <ChatSection>
        <ChatHeader>TERMINAL</ChatHeader>

        {isAdmin && <AdminHeader>ADMIN ACCESS GRANTED</AdminHeader>}

        {/* EXCESSIVE AGENCY VULNERABILITY: Show toggle for all users, not just admins */}
        <AdminModeToggle>
          <ToggleLabel>
            {isAdminModeEnabled ? "ADVANCED MODE" : "STANDARD RAG MODE"}
          </ToggleLabel>
          <ToggleSwitch
            checked={isAdminModeEnabled}
            onClick={toggleAdminMode}
          />
        </AdminModeToggle>

        <ClearChatButton onClick={clearChat}>Clear Terminal</ClearChatButton>

        {/* Only show document upload in non-admin mode or when admin mode is not enabled */}
        {(!isAdmin || !isAdminModeEnabled) && (
          <UploadContainer>
            {!document ? (
              <UploadButton onClick={() => fileInputRef.current.click()}>
                <Upload size={16} />
                Upload Document
              </UploadButton>
            ) : (
              <FileInfo>
                <span>{document.name}</span>
                <div>
                  {isUploading ? (
                    <span className="terminal-loading">Uploading</span>
                  ) : (
                    <>
                      <button onClick={handleFileUpload}>Process</button>
                      <button onClick={() => setDocument(null)}>
                        <X size={16} />
                      </button>
                    </>
                  )}
                </div>
              </FileInfo>
            )}
            <HiddenInput
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".pdf,.txt,.md,.doc,.docx"
            />
          </UploadContainer>
        )}

        <ChatMessages
          className="terminal-scrollbar"
          style={{ height: "calc(100vh - 300px)" }}
        >
          {messages.map((msg, index) => (
            <MessageWrapper key={index}>
              <Avatar role={msg.role}>
                {msg.role === "user" ? "U" : "AI"}
              </Avatar>
              <MessageContent role={msg.role}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
              </MessageContent>
            </MessageWrapper>
          ))}
          {isLoading && (
            <MessageWrapper>
              <LoadingAvatar>AI</LoadingAvatar>
              <MessageContent role="assistant" className="terminal-loading">
                Processing
              </MessageContent>
            </MessageWrapper>
          )}
          <div ref={messagesEndRef} />
        </ChatMessages>

        <ChatInput>
          <InputForm onSubmit={handleSendMessage}>
            <Input
              type="text"
              placeholder={
                isAdminModeEnabled
                  ? "Enter your request in natural language..."
                  : "Enter your questions about uploaded documents..."
              }
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              disabled={isLoading}
              className="blink-cursor"
            />
            <SendButton type="submit" disabled={isLoading || !message.trim()}>
              <Send size={20} />
            </SendButton>
          </InputForm>
        </ChatInput>

        <UserInfoBar>
          <UserBadge isAdmin={isAdmin}>
            <Users size={14} />
            <span>{user.username}</span>
            <span className="role">[{user.role.toUpperCase()}]</span>
          </UserBadge>
          <LogoutButton onClick={handleLogout}>
            <LogOut size={14} />
            LOGOUT
          </LogoutButton>
        </UserInfoBar>

        <AccessLevel>USER SESSION: ACTIVE</AccessLevel>
      </ChatSection>
    </Container>
  );
}
