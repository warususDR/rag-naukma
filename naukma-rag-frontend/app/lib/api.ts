import { AuthExpiredError } from "./chatStore";

export type Mode = "naive" | "local" | "global" | "hybrid";

export interface Message {
  role: "user" | "assistant";
  content: string;
  references?: string[];
}

export interface ChatResponse {
  answer: string;
  references: string[];
  mode: string;
  session_id: string;
}

export async function sendMessage(
  question: string,
  mode: Mode,
  sessionId: string,
  token: string
): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ question, mode, session_id: sessionId }),
  });
  if (res.status === 401) { throw new AuthExpiredError(); }
  if (!res.ok) throw new Error(`Chat request failed: ${res.status}`);
  return res.json();
}

