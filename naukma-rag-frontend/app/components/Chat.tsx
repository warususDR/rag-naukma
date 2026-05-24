import { useState, useRef, useEffect, useCallback } from "react";
import { googleLogout } from "@react-oauth/google";
import { toast } from "sonner";
import { useAuth } from "./AuthProvider";
import logoUrl from "../assets/logo.svg";
import { type Mode, type Message, sendMessage } from "../lib/api";
import {
  type ChatSession,
  AuthExpiredError,
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
  const auth = useAuth();

  const handleAuthError = useCallback((e: unknown) => {
    if (e instanceof AuthExpiredError) { auth.logout(); googleLogout(); }
    else throw e;
  }, [auth]);

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
    (async () => {
      const token = auth.credential!;
      const all = await getAllSessions(token);
      setSessions(all);
      const savedId = getActiveId();
      const found = savedId ? all.find((s) => s.id === savedId) : undefined;
      if (found) {
        setActiveSessionId(found.id);
        setMessages(found.messages);
        setMode(found.mode);
      } else if (all.length > 0) {
        loadSessionData(all[0]);
      }
    })().catch(handleAuthError);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  function loadSessionData(session: ChatSession) {
    setActiveSessionId(session.id);
    setActiveId(session.id);
    setMessages(session.messages);
    setMode(session.mode);
    setInput("");
  }

  async function refreshSessions() {
    const token = auth.credential!;
    setSessions(await getAllSessions(token));
  }

  const handleNewChat = async () => {
    try {
      const token = auth.credential!;
      const s = await createSession(mode, token);
      loadSessionData(s);
      await refreshSessions();
      setSidebarOpen(false);
    } catch (e) { handleAuthError(e); }
  };

  const handleSelectSession = async (id: string) => {
    if (id === activeSessionId) { setSidebarOpen(false); return; }
    try {
      const token = auth.credential!;
      const s = await getSession(id, token);
      loadSessionData(s);
      await refreshSessions();
      setSidebarOpen(false);
    } catch (e) { handleAuthError(e); }
  };

  const handleDeleteSession = async (id: string) => {
    try {
      const token = auth.credential!;
      await deleteSession(id, token);
      if (id === activeSessionId) {
        const remaining = await getAllSessions(token);
        if (remaining.length > 0) {
          loadSessionData(remaining[0]);
        } else {
          const s = await createSession(mode, token);
          loadSessionData(s);
        }
      }
      await refreshSessions();
    } catch (e) { handleAuthError(e); }
  };

  const handleModeChange = async (newMode: Mode) => {
    setMode(newMode);
    if (activeSessionId) {
      try {
        const token = auth.credential!;
        await updateSession(activeSessionId, { mode: newMode }, token);
      } catch (e) { handleAuthError(e); }
    }
  };

  const handleClear = async () => {
    try {
      const token = auth.credential!;
      if (activeSessionId) await deleteSession(activeSessionId, token);
      const s = await createSession(mode, token);
      loadSessionData(s);
      await refreshSessions();
    } catch (e) { handleAuthError(e); }
  };

  const handleSend = async () => {
    const question = input.trim();
    if (!question || loading) return;

    const token = auth.credential!;
    let sessionId = activeSessionId;
    if (!sessionId) {
      const s = await createSession(mode, token);
      sessionId = s.id;
      setActiveSessionId(s.id);
    }

    setInput("");
    const newMessages: Message[] = [...messages, { role: "user", content: question }];
    setMessages(newMessages);
    setLoading(true);

    try {
      const data = await sendMessage(question, mode, sessionId, token);
      const updated: Message[] = [...newMessages, { role: "assistant", content: data.answer, references: data.references }];
      setMessages(updated);
      await refreshSessions();
    } catch (e) {
      if (e instanceof AuthExpiredError) {
        auth.logout(); googleLogout();
      } else {
        setMessages([...newMessages, { role: "assistant", content: "Помилка з'єднання з сервером. Спробуйте ще раз." }]);
        toast.error("Не вдалось зв'язатись з сервером");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => { auth.logout(); googleLogout(); };
  const [profileOpen, setProfileOpen] = useState(false);

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
      <header className="flex-shrink-0 bg-naukma-navy text-white px-4 py-3 sm:py-4 shadow-lg">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 min-w-0">
              {/* Sidebar toggle */}
              <button
                onClick={() => setSidebarOpen(true)}
                className="flex-shrink-0 text-white/70 hover:text-white transition-colors cursor-pointer"
                title="Історія чатів"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 12h18M3 6h18M3 18h18" />
                </svg>
              </button>
              <div className="w-8 h-8 rounded-full overflow-hidden flex-shrink-0 hidden">
                <img src={logoUrl} alt="НаУКМА" className="w-full h-full object-cover" />
              </div>
              <div className="min-w-0">
                <h1 className="text-xl font-bold tracking-tight">
                  <span className="text-naukma-gold">НаУКМА</span> RAG
                </h1>
                <p className="text-xs text-white/60">Запитай про Могилянку</p>
              </div>
            </div>
            <div className="flex items-center gap-3 flex-shrink-0">
              {/* Mode selector */}
              <div className="hidden sm:block">
                <ModeSelector value={mode} onChange={handleModeChange} disabled={loading} />
              </div>
              <button
                onClick={handleClear}
                disabled={loading || messages.length === 0}
                className="text-white/60 hover:text-white transition-colors cursor-pointer
                  disabled:opacity-30 disabled:cursor-not-allowed"
                title="Очистити розмову"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
                </svg>
              </button>

              {/* Profile */}
              <div className="relative">
                <button
                  onClick={() => setProfileOpen((o) => !o)}
                  className="w-8 h-8 rounded-full overflow-hidden ring-2 ring-white/20 hover:ring-white/60 transition-all cursor-pointer flex-shrink-0"
                  title={auth.email ?? "Профіль"}
                >
                  {auth.picture ? (
                    <img src={auth.picture} alt="avatar" className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                  ) : (
                    <div className="w-full h-full bg-naukma-gold flex items-center justify-center text-naukma-navy font-bold text-sm">
                      {(auth.name ?? auth.email ?? "?")[0].toUpperCase()}
                    </div>
                  )}
                </button>

                {profileOpen && (
                  <>
                    {/* Backdrop */}
                    <div className="fixed inset-0 z-10" onClick={() => setProfileOpen(false)} />
                    {/* Dropdown */}
                    <div className="absolute right-0 top-10 z-20 w-64 bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden">
                      <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-100">
                        <div className="w-10 h-10 rounded-full overflow-hidden flex-shrink-0">
                          {auth.picture ? (
                            <img src={auth.picture} alt="avatar" className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                          ) : (
                            <div className="w-full h-full bg-naukma-gold flex items-center justify-center text-naukma-navy font-bold">
                              {(auth.name ?? auth.email ?? "?")[0].toUpperCase()}
                            </div>
                          )}
                        </div>
                        <div className="min-w-0">
                          {auth.name && <p className="text-sm font-semibold text-gray-900 truncate">{auth.name}</p>}
                          <p className="text-xs text-gray-500 truncate">{auth.email}</p>
                        </div>
                      </div>
                      <button
                        onClick={() => { setProfileOpen(false); handleLogout(); }}
                        className="w-full flex items-center gap-2 px-4 py-3 text-sm text-red-600 hover:bg-red-50 transition-colors cursor-pointer"
                      >
                        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5-5-5M21 12H9" />
                        </svg>
                        Вийти з акаунту
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
          {/* Mode selector */}
          <div className="mt-2 sm:hidden">
            <ModeSelector value={mode} onChange={handleModeChange} disabled={loading} />
          </div>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center h-full text-center pt-24">
              <div className="w-20 h-20 rounded-full overflow-hidden mb-6">
                <img src={logoUrl} alt="НаУКМА" className="w-full h-full object-cover" />
              </div>
              <h2 className="text-lg font-semibold text-naukma-navy mb-2">
                Вітаю! Я — асистент НаУКМА.
              </h2>
              <p className="text-sm text-gray-500 max-w-md">
                Запитайте мене про Національний університет «Києво-Могилянська академія» —
                накази, викладачі, факультети, програми, історію, правила вступу тощо.
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
