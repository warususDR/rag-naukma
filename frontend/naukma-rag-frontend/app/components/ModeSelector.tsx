import type { Mode } from "../lib/api";

const MODE_LABELS: Record<Mode, string> = {
  naive: "Naive",
  local: "Local",
  global: "Global",
  hybrid: "Hybrid",
};

interface ModeSelectorProps {
  value: Mode;
  onChange: (mode: Mode) => void;
  disabled?: boolean;
}

export function ModeSelector({ value, onChange, disabled }: ModeSelectorProps) {
  const modes: Mode[] = ["naive", "local", "global", "hybrid"];

  return (
    <div className="flex gap-1">
      {modes.map((mode) => (
        <button
          key={mode}
          onClick={() => onChange(mode)}
          disabled={disabled}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors cursor-pointer
            ${
              value === mode
                ? "bg-naukma-navy text-naukma-gold shadow-md"
                : "bg-white/60 text-naukma-navy hover:bg-white border border-naukma-navy/15"
            }
            disabled:opacity-50 disabled:cursor-not-allowed`}
        >
          {MODE_LABELS[mode]}
        </button>
      ))}
    </div>
  );
}
