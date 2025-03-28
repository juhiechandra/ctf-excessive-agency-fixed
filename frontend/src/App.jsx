import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider } from "styled-components";
import { MantineProvider } from "@mantine/core";
import { theme } from "./styles/theme";
import ChatPage from "./pages/ChatPage";
import IntroPage from "./pages/IntroPage";

function App() {
  return (
    <ThemeProvider theme={theme}>
      <MantineProvider withGlobalStyles withNormalizeCSS>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<IntroPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </MantineProvider>
    </ThemeProvider>
  );
}

export default App;
