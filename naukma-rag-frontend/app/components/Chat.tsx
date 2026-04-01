import { useState, useRef, useEffect, useCallback } from "react";
import { type Mode, type Message, sendMessage, clearHistory as clearServerHistory } from "../lib/api";
import {
  type ChatSession,
  getAllSessions,
  getSession,
  createSession,
  updateSession,
  deleteSession,
  getActiveId,
  setActiveId,
} from "../lib/chatStore";
import { ModeSelector } from "./ModeSelector";
import { MessageBubble } from "./MessageBubble";
import { ChatInput } from "./ChatInput";
import { Sidebar } from "./Sidebar";

export function Chat() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [mode, setMode] = useState<Mode>("hybrid");

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    const all = getAllSessions();
    setSessions(all);
    const savedId = getActiveId();
    const found = savedId ? all.find((s) => s.id === savedId) : undefined;
    if (found) {
      setActiveSessionId(found.id);
      setMessages(found.messages);
      setMode(found.mode);
    } else if (all.length > 0) {
      loadSession(all[0]);
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  function refreshSessions() {
    setSessions(getAllSessions());
  }

  function loadSession(session: ChatSession) {
    setActiveSessionId(session.id);
    setActiveId(session.id);
    setMessages(session.messages);
    setMode(session.mode);
    setInput("");
  }

  const handleNewChat = () => {
    if (activeSessionId) {
      updateSession(activeSessionId, { messages, mode });
    }
    const s = createSession(mode);
    refreshSessions();
    loadSession(s);
    clearServerHistory().catch(() => {});
    setSidebarOpen(false);
  };

  const handleSelectSession = (id: string) => {
    if (id === activeSessionId) {
      setSidebarOpen(false);
      return;
    }
    if (activeSessionId) {
      updateSession(activeSessionId, { messages, mode });
    }
    const s = getSession(id);
    if (s) {
      loadSession(s);
      clearServerHistory().catch(() => {});
    }
    refreshSessions();
    setSidebarOpen(false);
  };

  const handleDeleteSession = (id: string) => {
    deleteSession(id);
    if (id === activeSessionId) {
      const remaining = getAllSessions();
      if (remaining.length > 0) {
        loadSession(remaining[0]);
      } else {
        setActiveSessionId(null);
        setActiveId(null);
        setMessages([]);
      }
    }
    refreshSessions();
  };

  const handleModeChange = (newMode: Mode) => {
    setMode(newMode);
    if (activeSessionId) {
      updateSession(activeSessionId, { mode: newMode });
    }
  };

  const handleSend = async () => {
    const question = input.trim();
    if (!question || loading) return;

    let sessionId = activeSessionId;
    if (!sessionId) {
      const s = createSession(mode);
      sessionId = s.id;
      setActiveSessionId(s.id);
      refreshSessions();
    }

    setInput("");
    const newMessages: Message[] = [...messages, { role: "user", content: question }];
    setMessages(newMessages);
    setLoading(true);

    try {
      const data = await sendMessage(question, mode);
      const updated: Message[] = [...newMessages, { role: "assistant", content: data.answer }];
      setMessages(updated);
      updateSession(sessionId, { messages: updated, mode });
      refreshSessions();
    } catch {
      const updated: Message[] = [
        ...newMessages,
        { role: "assistant", content: "Помилка з'єднання з сервером. Спробуйте ще раз." },
      ];
      setMessages(updated);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (activeSessionId) {
      deleteSession(activeSessionId);
    }
    const s = createSession(mode);
    loadSession(s);
    refreshSessions();
    clearServerHistory().catch(() => {});
  };

  return (
    <div className="flex flex-col h-screen">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        sessions={sessions}
        activeId={activeSessionId}
        onSelect={handleSelectSession}
        onNew={handleNewChat}
        onDelete={handleDeleteSession}
      />

      {/* Header */}
      <header className="flex-shrink-0 bg-naukma-navy text-white px-4 py-4 shadow-lg">
        <div className="max-w-4xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            {/* Sidebar toggle */}
            <button
              onClick={() => setSidebarOpen(true)}
              className="text-white/70 hover:text-white transition-colors cursor-pointer"
              title="Історія чатів"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18" />
              </svg>
            </button>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                <span className="text-naukma-gold">НаУКМА</span> RAG
              </h1>
              <p className="text-xs text-white/60">Запитай про Могилянку</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <ModeSelector value={mode} onChange={handleModeChange} disabled={loading} />
            <button
              onClick={handleClear}
              disabled={loading || messages.length === 0}
              className="text-sm text-white/60 hover:text-white transition-colors cursor-pointer
                disabled:opacity-30 disabled:cursor-not-allowed"
              title="Очистити розмову"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
              </svg>
            </button>
          </div>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center h-full text-center pt-24">
              <div className="w-20 h-20 rounded-full bg-naukma-navy/10 flex items-center justify-center mb-6">
                <svg className="w-10 h-10 text-naukma-navy/40" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 3L1 9l4 2.18v6L12 21l7-3.82v-6l2-1.09V17h2V9L12 3zm6.82 6L12 12.72 5.18 9 12 5.28 18.82 9zM17 15.99l-5 2.73-5-2.73v-3.72L12 15l5-2.73v3.72z" />
                </svg>
              </div>
              <h2 className="text-lg font-semibold text-naukma-navy mb-2">
                Вітаю! Я — асистент НаУКМА.
              </h2>
              <p className="text-sm text-gray-500 max-w-md">
                Запитайте мене про Національний університет «Києво-Могилянська академія» —
                факультети, програми, історію, правила вступу тощо.
              </p>
            </div>
          )}
          {messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-naukma-navy/10 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 bg-naukma-navy/30 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 bg-naukma-navy/30 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 bg-naukma-navy/30 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </main>

      {/* Input */}
      <footer className="flex-shrink-0 border-t border-naukma-navy/10 bg-white/80 backdrop-blur px-4 py-4">
        <div className="max-w-4xl mx-auto">
          <ChatInput value={input} onChange={setInput} onSend={handleSend} disabled={loading} />
        </div>
      </footer>
    </div>
  );
}
