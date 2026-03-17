export type Mode = "naive" | "local" | "global" | "hybrid";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  answer: string;
  mode: string;
}

export async function sendMessage(question: string, mode: Mode): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode }),
  });
  if (!res.ok) throw new Error(`Chat request failed: ${res.status}`);
  return res.json();
}

export async function getHistory(): Promise<Message[]> {
  const res = await fetch("/api/history");
  if (!res.ok) throw new Error(`History request failed: ${res.status}`);
  const data = await res.json();
  return data.history ?? [];
}

export async function clearHistory(): Promise<void> {
  const res = await fetch("/api/clear", { method: "POST" });
  if (!res.ok) throw new Error(`Clear request failed: ${res.status}`);
}
