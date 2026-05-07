import type { Route } from "./+types/home";
import { Navigate } from "react-router";
import { useAuth } from "../components/AuthProvider";
import { Chat } from "../components/Chat";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "НаУКМА RAG — Запитай про Могилянку" },
    { name: "description", content: "Чат-бот з базою знань НаУКМА" },
  ];
}

export default function Home() {
  const { credential } = useAuth();
  if (!credential) return <Navigate to="/login" replace />;
  return <Chat />;
}

