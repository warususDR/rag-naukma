import type { Message, Mode } from "./api";

export interface ChatSession {
  id: string;
  title: string;
  mode: Mode;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

const ACTIVE_KEY = "naukma-rag-active-chat";

export function getActiveId(): string | null {
  return localStorage.getItem(ACTIVE_KEY);
}

export function setActiveId(id: string | null) {
  if (id) localStorage.setItem(ACTIVE_KEY, id);
  else localStorage.removeItem(ACTIVE_KEY);
}

function authHeaders(token: string) {
  return { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };
}

export class AuthExpiredError extends Error {
  constructor() { super("AUTH_EXPIRED"); }
}

function checkAuth(res: Response) {
  if (res.status === 401) throw new AuthExpiredError();
}

export async function getAllSessions(token: string): Promise<ChatSession[]> {
  const res = await fetch("/api/sessions", { headers: authHeaders(token) });
  checkAuth(res);
  if (!res.ok) throw new Error("Failed to fetch sessions");
  return res.json();
}

export async function getSession(id: string, token: string): Promise<ChatSession> {
  const res = await fetch(`/api/sessions/${id}`, { headers: authHeaders(token) });
  checkAuth(res);
  if (!res.ok) throw new Error("Failed to fetch session");
  return res.json();
}

export async function createSession(mode: Mode, token: string): Promise<ChatSession> {
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ mode }),
  });
  checkAuth(res);
  if (!res.ok) throw new Error("Failed to create session");
  return res.json();
}

export async function updateSession(
  id: string,
  patch: Partial<{ title: string; mode: Mode }>,
  token: string
): Promise<void> {
  await fetch(`/api/sessions/${id}`, {
    method: "PATCH",
    headers: authHeaders(token),
    body: JSON.stringify(patch),
  });
}

export async function deleteSession(id: string, token: string): Promise<void> {
  await fetch(`/api/sessions/${id}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
}

