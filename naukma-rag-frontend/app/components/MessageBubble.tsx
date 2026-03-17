import Markdown from "react-markdown";
import type { Message } from "../lib/api";

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[75%] rounded-2xl px-4 py-3 text-[15px] leading-relaxed
          ${
            isUser
              ? "bg-naukma-navy text-white rounded-br-md whitespace-pre-wrap"
              : "bg-white border border-naukma-navy/10 text-gray-800 rounded-bl-md shadow-sm prose prose-sm prose-gray max-w-none"
          }`}
      >
        {isUser ? (
          message.content
        ) : (
          <Markdown
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              ul: ({ children }) => <ul className="list-disc pl-4 mb-2">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal pl-4 mb-2">{children}</ol>,
              li: ({ children }) => <li className="mb-0.5">{children}</li>,
              strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
              h3: ({ children }) => <h3 className="font-semibold text-base mt-3 mb-1">{children}</h3>,
              h4: ({ children }) => <h4 className="font-semibold text-sm mt-2 mb-1">{children}</h4>,
              code: ({ children }) => (
                <code className="bg-gray-100 rounded px-1 py-0.5 text-sm font-mono">{children}</code>
              ),
            }}
          >
            {message.content}
          </Markdown>
        )}
      </div>
    </div>
  );
}
