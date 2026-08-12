import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes, useNavigate } from "react-router-dom";
import App from "./App";
import { Landing } from "./components/Landing";
import { Login } from "./components/Login";
import { RequireAuth } from "./components/RequireAuth";
import { Signup } from "./components/Signup";
import { AuthProvider } from "./context/AuthContext";
import "./styles.css";

function LandingRoute() {
  const navigate = useNavigate();
  return <Landing onStart={() => navigate("/app")} />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/" element={<LandingRoute />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route
            path="/app"
            element={
              <RequireAuth>
                <App />
              </RequireAuth>
            }
          />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
);
