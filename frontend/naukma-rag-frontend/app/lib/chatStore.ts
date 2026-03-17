import type { Message, Mode } from "./api";

export interface ChatSession {
  id: string;
  title: string;
  mode: Mode;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = "naukma-rag-chats";
const ACTIVE_KEY = "naukma-rag-active-chat";

function readAll(): ChatSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function writeAll(sessions: ChatSession[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

export function getActiveId(): string | null {
  return localStorage.getItem(ACTIVE_KEY);
}

export function setActiveId(id: string | null) {
  if (id) localStorage.setItem(ACTIVE_KEY, id);
  else localStorage.removeItem(ACTIVE_KEY);
}

export function getAllSessions(): ChatSession[] {
  return readAll().sort((a, b) => b.updatedAt - a.updatedAt);
}

export function getSession(id: string): ChatSession | undefined {
  return readAll().find((s) => s.id === id);
}

export function createSession(mode: Mode = "hybrid"): ChatSession {
  const session: ChatSession = {
    id: crypto.randomUUID(),
    title: "Нова розмова",
    mode,
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  const all = readAll();
  all.push(session);
  writeAll(all);
  setActiveId(session.id);
  return session;
}

export function updateSession(id: string, patch: Partial<Pick<ChatSession, "title" | "mode" | "messages">>) {
  const all = readAll();
  const idx = all.findIndex((s) => s.id === id);
  if (idx === -1) return;
  Object.assign(all[idx], patch, { updatedAt: Date.now() });
  // Auto-title from first user message
  if (patch.messages && all[idx].title === "Нова розмова") {
    const firstUser = patch.messages.find((m) => m.role === "user");
    if (firstUser) {
      all[idx].title = firstUser.content.slice(0, 50) + (firstUser.content.length > 50 ? "…" : "");
    }
  }
  writeAll(all);
}

export function deleteSession(id: string) {
  const all = readAll().filter((s) => s.id !== id);
  writeAll(all);
  if (getActiveId() === id) setActiveId(null);
}
