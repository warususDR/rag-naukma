import { GoogleOAuthProvider } from "@react-oauth/google";
import { GOOGLE_CLIENT_ID } from "../lib/authConfig";
import { createContext, useContext, useState } from "react";

interface AuthContextValue {
  credential: string | null;
  email: string | null;
  name: string | null;
  picture: string | null;
  login: (credential: string, email: string, name: string, picture: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue>({
  credential: null,
  email: null,
  name: null,
  picture: null,
  login: () => {},
  logout: () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

function InnerAuthProvider({ children }: { children: React.ReactNode }) {
  const [credential, setCredential] = useState<string | null>(
    () => localStorage.getItem("google_credential")
  );
  const [email, setEmail] = useState<string | null>(
    () => localStorage.getItem("google_email")
  );
  const [name, setName] = useState<string | null>(
    () => localStorage.getItem("google_name")
  );
  const [picture, setPicture] = useState<string | null>(
    () => localStorage.getItem("google_picture")
  );

  const login = (cred: string, em: string, nm: string, pic: string) => {
    localStorage.setItem("google_credential", cred);
    localStorage.setItem("google_email", em);
    localStorage.setItem("google_name", nm);
    localStorage.setItem("google_picture", pic);
    setCredential(cred);
    setEmail(em);
    setName(nm);
    setPicture(pic);
  };

  const logout = () => {
    localStorage.removeItem("google_credential");
    localStorage.removeItem("google_email");
    localStorage.removeItem("google_name");
    localStorage.removeItem("google_picture");
    setCredential(null);
    setEmail(null);
    setName(null);
    setPicture(null);
  };

  return (
    <AuthContext.Provider value={{ credential, email, name, picture, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID} locale="uk">
      <InnerAuthProvider>{children}</InnerAuthProvider>
    </GoogleOAuthProvider>
  );
}
