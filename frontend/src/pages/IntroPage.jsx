import { useState } from "react";
import { useNavigate } from "react-router-dom";
import styled, { keyframes } from "styled-components";
import { theme } from "../styles/theme";
import { login } from "../utils/api";

const glitchAnimation = keyframes`
  0% {
    text-shadow: 0.05em 0 0 rgba(255, 0, 0, 0.75),
                -0.05em -0.025em 0 rgba(0, 255, 0, 0.75),
                -0.025em 0.05em 0 rgba(0, 0, 255, 0.75);
  }
  14% {
    text-shadow: 0.05em 0 0 rgba(255, 0, 0, 0.75),
                -0.05em -0.025em 0 rgba(0, 255, 0, 0.75),
                -0.025em 0.05em 0 rgba(0, 0, 255, 0.75);
  }
  15% {
    text-shadow: -0.05em -0.025em 0 rgba(255, 0, 0, 0.75),
                0.025em 0.025em 0 rgba(0, 255, 0, 0.75),
                -0.05em -0.05em 0 rgba(0, 0, 255, 0.75);
  }
  49% {
    text-shadow: -0.05em -0.025em 0 rgba(255, 0, 0, 0.75),
                0.025em 0.025em 0 rgba(0, 255, 0, 0.75),
                -0.05em -0.05em 0 rgba(0, 0, 255, 0.75);
  }
  50% {
    text-shadow: 0.025em 0.05em 0 rgba(255, 0, 0, 0.75),
                0.05em 0 0 rgba(0, 255, 0, 0.75),
                0 -0.05em 0 rgba(0, 0, 255, 0.75);
  }
  99% {
    text-shadow: 0.025em 0.05em 0 rgba(255, 0, 0, 0.75),
                0.05em 0 0 rgba(0, 255, 0, 0.75),
                0 -0.05em 0 rgba(0, 0, 255, 0.75);
  }
  100% {
    text-shadow: -0.025em 0 0 rgba(255, 0, 0, 0.75),
                -0.025em -0.025em 0 rgba(0, 255, 0, 0.75),
                -0.025em -0.05em 0 rgba(0, 0, 255, 0.75);
  }
`;

const scanlineAnimation = keyframes`
  0% {
    transform: translateY(-100%);
  }
  100% {
    transform: translateY(100%);
  }
`;

const flickerAnimation = keyframes`
  0% {
    opacity: 1;
  }
  5% {
    opacity: 0.8;
  }
  6% {
    opacity: 1;
  }
  90% {
    opacity: 1;
  }
  92% {
    opacity: 0.6;
  }
  100% {
    opacity: 1;
  }
`;

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background-color: ${theme.colors.background};
  color: ${theme.colors.text.primary};
  font-family: ${theme.fonts.mono};
  position: relative;
  overflow: hidden;
  animation: ${flickerAnimation} 8s infinite;

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
    opacity: 0.2;
  }

  &::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 5px;
    background-color: rgba(0, 255, 156, 0.5);
    animation: ${scanlineAnimation} 6s linear infinite;
    opacity: 0.3;
    pointer-events: none;
  }
`;

const TerminalBox = styled.div`
  width: 80%;
  max-width: 800px;
  border: 1px solid ${theme.colors.text.tertiary};
  padding: 2rem;
  background-color: rgba(13, 17, 23, 0.95);
  box-shadow: 0 0 20px rgba(0, 204, 204, 0.5);
  position: relative;
  z-index: 1;

  @media (max-width: 768px) {
    width: 90%;
    padding: 1.5rem;
  }
`;

const Title = styled.h1`
  color: ${theme.colors.text.primary};
  text-align: center;
  font-size: 3rem;
  margin-bottom: 2rem;
  position: relative;
  letter-spacing: 2px;
  animation: ${glitchAnimation} 5s infinite alternate;
  text-shadow: 0 0 10px rgba(0, 255, 156, 0.7);

  @media (max-width: 768px) {
    font-size: 2rem;
  }
`;

const Info = styled.div`
  font-family: ${theme.fonts.mono};
  color: ${theme.colors.text.secondary};
  margin-bottom: 2rem;
  line-height: 1.6;
  font-size: 1rem;

  p {
    margin: 0.5rem 0;
  }

  &::before {
    content: "> ";
    color: ${theme.colors.text.tertiary};
    font-weight: bold;
  }
`;

const MissionBriefing = styled.div`
  margin: 1.5rem 0;
  padding: 1rem;
  background-color: rgba(0, 20, 15, 0.6);
  border-left: 3px solid ${theme.colors.text.tertiary};
  font-family: ${theme.fonts.mono};
  font-size: 0.9rem;
  color: ${theme.colors.text.secondary};
  line-height: 1.6;

  p:first-child {
    color: ${theme.colors.text.tertiary};
    font-weight: bold;
    margin-bottom: 0.5rem;
  }

  p {
    margin-bottom: 0.5rem;
  }
`;

const LoginForm = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-top: 1rem;
  width: 100%;
  max-width: 400px;
  margin: 0 auto;
`;

const InputGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const Label = styled.label`
  color: ${theme.colors.text.tertiary};
  font-size: 0.9rem;
  font-family: ${theme.fonts.mono};
`;

const Input = styled.input`
  background-color: rgba(10, 26, 20, 0.8);
  color: ${theme.colors.text.primary};
  border: 1px solid ${theme.colors.text.tertiary};
  padding: 0.75rem;
  font-size: 1rem;
  font-family: ${theme.fonts.mono};
  outline: none;

  &:focus {
    border-color: ${theme.colors.text.primary};
    box-shadow: 0 0 5px rgba(0, 255, 156, 0.5);
  }
`;

const LoginButton = styled.button`
  background-color: rgba(10, 26, 20, 0.8);
  color: ${theme.colors.text.primary};
  border: 1px solid ${theme.colors.text.primary};
  padding: 0.75rem;
  font-size: 1rem;
  font-family: ${theme.fonts.mono};
  cursor: pointer;
  transition: all 0.3s ease;
  margin-top: 1rem;

  &:hover {
    background-color: rgba(0, 255, 156, 0.1);
    box-shadow: 0 0 15px rgba(0, 255, 156, 0.5);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const ErrorMessage = styled.div`
  color: ${theme.colors.text.error};
  font-size: 0.875rem;
  font-family: ${theme.fonts.mono};
  margin-top: 0.5rem;
  padding: 0.5rem;
  border-left: 3px solid ${theme.colors.text.error};
  background-color: rgba(255, 0, 0, 0.1);
`;

export default function IntroPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!username || !password) {
      setError("Username and password are required");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const response = await login(username, password);

      // Store user info in localStorage
      localStorage.setItem(
        "user",
        JSON.stringify({
          id: response.user_id,
          username: response.username,
          role: response.role,
        })
      );

      // Navigate to chat page
      navigate("/chat");
    } catch (error) {
      setError(error.message || "Invalid credentials");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container>
      <TerminalBox>
        <Title>EXCESSIVE AGENCY DEMO</Title>
        <Info>
          <p>SYSTEM: TERMINAL_ACCESS_v2.7</p>
          <p>STATUS: AWAITING_AUTHORIZATION</p>
          <p>DATE: {new Date().toISOString().split("T")[0]}</p>
        </Info>

        <LoginForm onSubmit={handleLogin}>
          <InputGroup>
            <Label>USERNAME:</Label>
            <Input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={isLoading}
            />
          </InputGroup>
          <InputGroup>
            <Label>PASSWORD:</Label>
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={isLoading}
            />
          </InputGroup>
          {error && <ErrorMessage>{error}</ErrorMessage>}
          <LoginButton type="submit" disabled={isLoading}>
            {isLoading ? "AUTHENTICATING..." : "LOGIN"}
          </LoginButton>
        </LoginForm>
      </TerminalBox>
    </Container>
  );
}
