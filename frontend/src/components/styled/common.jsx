import styled from "styled-components";

export const Container = styled.div`
  width: 100%;
  min-height: 100vh;
  background: #000;
  color: #fff;
`;

export const Card = styled.div`
  background: #111;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 2rem;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
`;

export const Button = styled.button`
  background: #fff;
  color: #000;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: #f3f3f3;
  }

  &.secondary {
    background: transparent;
    border: 1px solid #333;
    color: #fff;

    &:hover {
      border-color: #666;
    }
  }
`;

export const Input = styled.input`
  width: 100%;
  padding: 0.75rem 1rem;
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 6px;
  color: #fff;
  margin-bottom: 1rem;

  &:focus {
    outline: none;
    border-color: #666;
  }
`;
