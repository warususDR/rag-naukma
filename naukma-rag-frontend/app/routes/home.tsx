import type { Route } from "./+types/home";
import { Chat } from "../components/Chat";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "НаУКМА RAG — Запитай про Могилянку" },
    { name: "description", content: "Чат-бот з базою знань НаУКМА" },
  ];
}

export default function Home() {
  return <Chat />;
}
