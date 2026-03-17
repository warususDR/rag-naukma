import type { ChatSession } from "../lib/chatStore";

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  sessions: ChatSession[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

export function Sidebar({ open, onClose, sessions, activeId, onSelect, onNew, onDelete }: SidebarProps) {
  return (
    <>
      {/* Backdrop */}
      {open && (
        <div className="fixed inset-0 bg-black/30 z-40 lg:hidden" onClick={onClose} />
      )}

      {/* Panel */}
      <aside
        className={`fixed top-0 left-0 h-full w-72 bg-naukma-dark text-white z-50
          transform transition-transform duration-200 ease-in-out flex flex-col
          ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-white/10">
          <h2 className="font-semibold text-sm text-white/80">Історія чатів</h2>
          <button
            onClick={onClose}
            className="text-white/50 hover:text-white transition-colors cursor-pointer"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* New chat button */}
        <div className="px-3 py-3">
          <button
            onClick={onNew}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg border border-white/15
              text-sm text-white/80 hover:bg-white/10 transition-colors cursor-pointer"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12h14" />
            </svg>
            Нова розмова
          </button>
        </div>

        {/* Session list */}
        <nav className="flex-1 overflow-y-auto px-3 pb-3 space-y-1">
          {sessions.length === 0 && (
            <p className="text-xs text-white/30 text-center mt-8">Ще немає розмов</p>
          )}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`group flex items-center gap-2 rounded-lg px-3 py-2.5 text-sm cursor-pointer transition-colors
                ${s.id === activeId ? "bg-white/15 text-white" : "text-white/60 hover:bg-white/8 hover:text-white/90"}`}
              onClick={() => onSelect(s.id)}
            >
              <svg className="w-4 h-4 flex-shrink-0 opacity-50" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
              </svg>
              <span className="flex-1 truncate">{s.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(s.id);
                }}
                className="opacity-0 group-hover:opacity-100 text-white/40 hover:text-red-400 transition-all cursor-pointer"
                title="Видалити"
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}
        </nav>
      </aside>
    </>
  );
}
