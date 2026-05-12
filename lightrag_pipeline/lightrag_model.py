import json
import logging
import asyncio
import re
from typing import List, Dict, Optional
import nest_asyncio
import chromadb
from functools import partial
from chromadb.config import Settings
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from pathlib import Path

nest_asyncio.apply()

logger = logging.getLogger(__name__)

NAUKMA_SYSTEM_PROMPT = (
    'Ти - помічник для відповідей на питання про НаУКМА '
    '(Національний університет "Києво-Могилянська академія"). '
    "Перелічуй лише те, що прямо підтверджено в наданій інформації. "
    "Вигадування суворо заборонено, у випадку, якщо не знаєш чогось, повідом про це. "
    "Ти маєш надавати перевагу українській мові у своїх відповідях, уникай англійської мови. "
    "Надавай пріоритет наданому контексту, а не історії повідомлень, якщо є конфлікт між ними. "
    "Ніколи не цитуй і не відтворюй технічні дані у відповіді: JSON, XML, фрагменти коду, "
    "назви полів (entity1, entity2, description тощо) та будь-яку інформацію з Knowledge Graph — використовуй їх лише для формування "
    "відповіді як контекст."
)

_GREETING_PATTERN = re.compile(
    r"^(привіт|здоров|вітаю|добрий\s+(день|ранок|вечір)|hi|hello|hey|дякую|спасибі|бувай|до побачення)[\s!?.]*$",
    re.IGNORECASE,
)


def create_embedding_func(model_name: str, embedding_dim: int, max_token_size: int, host: str = "http://localhost:11434") -> EmbeddingFunc:
    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        model_name=model_name,
        func=partial(ollama_embed.func, embed_model=model_name, host=host)
    )


class LightRAGModel:
    def __init__(
        self,
        embedding_model: str,
        llm_model_name: str,
        chroma_directory: str = "./chroma_db",
        chroma_collection: str = "naukma_documents_no_chunks",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        num_ctx: int = 30000,
        kv_storage: str = "PGKVStorage",
        vector_storage: str = "PGVectorStorage",
        graph_storage: str = "Neo4JStorage",
        doc_status_storage: str = "PGDocStatusStorage",
        ollama_host: str = "http://localhost:11434"
    ):
        self._llm_model_name = llm_model_name
        self._ollama_host = ollama_host
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_ctx = num_ctx

        self.embedding_func = create_embedding_func(model_name=embedding_model, embedding_dim=1024, max_token_size=8192, host=ollama_host)

        self.rag = LightRAG(
            llm_model_func=self._llm_model_func,
            llm_model_name=self._llm_model_name,
            llm_model_kwargs={
                "options": {
                    "temperature": self._temperature,
                    "repeat_penalty": 1.1,
                    "repeat_last_n": 64,
                    "num_ctx": self._num_ctx,
                },
            },
            embedding_func=self.embedding_func,
            summary_max_tokens=1600,
            addon_params={
                "language": "Ukrainian",
                "entity_types": ["Особа", "Організація", "Місце", "Подія", "Концепція", "Документ", "Дата", "Артефакт"],
            },
            kv_storage=kv_storage,
            vector_storage=vector_storage,
            graph_storage=graph_storage,
            doc_status_storage=doc_status_storage,
        )

        self._loop = asyncio.get_event_loop()
        self._loop.run_until_complete(self.rag.initialize_storages())

        logger.info(f"Connecting to Chroma DB at {chroma_directory}")
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        try:
            self.chroma_collection = self.chroma_client.get_collection(name=chroma_collection)
            logger.info(f"Connected to collection '{chroma_collection}' with {self.chroma_collection.count()} documents")
        except Exception as e:
            logger.warning(f"Could not connect to Chroma collection: {e}")
            self.chroma_collection = None

        self._history: List[Dict[str, str]] = []

        logger.info("LightRAG model initialized successfully")


    async def _llm_model_func(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: list = [],
        **kwargs
    ) -> str:
        # Strip <SEP> delimiters that leak from graph descriptions
        prompt = prompt.replace("<SEP>", " ")
        if system_prompt:
            system_prompt = system_prompt.replace("<SEP>", " ")

        return await ollama_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            host=self._ollama_host,
            **kwargs
        )


    def insert_documents(self, texts: List[str], file_paths: List[str] = None):
        logger.info(f"Inserting {len(texts)} documents into LightRAG...")
        self.rag.insert(texts, file_paths=file_paths)
        logger.info("Documents inserted successfully")


    def insert_from_text_dir(self, text_dir: str, max_files: Optional[int] = None, batch_size: int = 50):
        txt_files = sorted(Path(text_dir).glob("*.txt"))
        if max_files:
            txt_files = txt_files[:max_files]
        total = len(txt_files)
        logger.info(f"Found {total} .txt files in {text_dir}")
        for i in range(0, total, batch_size):
            batch = txt_files[i:i + batch_size]
            texts, paths = [], []
            for txt_path in batch:
                try:
                    texts.append(txt_path.read_text(encoding="utf-8"))
                    paths.append(str(txt_path))
                except Exception as e:
                    logger.error(f"Could not read {txt_path.name}: {e}")
            if texts:
                self.insert_documents(texts, file_paths=paths)
                logger.info(f"Ingested {min(i + batch_size, total)}/{total} files")


    def insert_from_json(self, json_path: str, max_files: Optional[int] = None, batch_size: int = 50):
        with open(json_path, encoding="utf-8") as f:
            entries = json.load(f)

        if max_files:
            entries = entries[:max_files]

        total = len(entries)
        logger.info(f"Found {total} documents in {json_path}")

        for i in range(0, total, batch_size):
            batch = entries[i:i + batch_size]
            texts, paths = [], []
            for entry in batch:
                text = entry.get("text", "").strip()
                if not text:
                    logger.warning(f"Skipping empty entry: {entry.get('filename', '?')}")
                    continue
                texts.append(text)
                paths.append(entry.get("filepath", entry.get("filename", "")))
            if texts:
                self.insert_documents(texts, file_paths=paths)
                logger.info(f"Ingested {min(i + batch_size, total)}/{total} documents")

    def load_from_chroma(self, max_documents: Optional[int] = None, batch_size: int = 50):
        if self.chroma_collection is None:
            logger.warning("No Chroma collection available")
            return

        count = self.chroma_collection.count()
        limit = min(count, max_documents) if max_documents else count

        logger.info(f"Loading {limit} documents from Chroma DB...")

        for offset in range(0, limit, batch_size):
            batch_limit = min(batch_size, limit - offset)
            results = self.chroma_collection.get(
                limit=batch_limit,
                offset=offset,
                include=["documents"]
            )

            if results and results['documents']:
                self.insert_documents(results['documents'])
                logger.info(f"Processed {offset + len(results['documents'])}/{limit} documents")

    def query(self, question: str, mode: str = "hybrid", history: Optional[List[Dict[str, str]]] = None) -> Dict:
        logger.info(f"Query: {question} (mode: {mode})")

        if _GREETING_PATTERN.match(question.strip()):
            logger.info("Greeting detected — using bypass mode")
            effective_mode = "bypass"
        else:
            effective_mode = mode

        conversation_history = list(history) if history else []

        param = QueryParam(
            mode=effective_mode,
            top_k=10,
            conversation_history=conversation_history,
            user_prompt=NAUKMA_SYSTEM_PROMPT,
            enable_rerank=False
        )

        result = self._loop.run_until_complete(
            self.rag.aquery_llm(query=question, param=param)
        )

        response = (result.get("llm_response") or {}).get("content") or ""

        # Strip leading heading
        response = re.sub(r'^#{1,3}\s*\S.*\n+', '', response).strip()

        # Strip any trailing references
        refs_match = re.search(r'\n+#{1,3}\s*References\s*\n.*$', response, flags=re.DOTALL | re.IGNORECASE)
        if refs_match:
            response = response[:refs_match.start()].strip()

        # Strip inline citation labels
        response = re.sub(r'\s*\bReferences?:\s*\[[\d,\s]+\]', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\*{1,2}\s*\[[\d,\s]+\]\s*\*{1,2}', '', response)  # **[1]** or *[1]*
        response = re.sub(r'\s*\[[\d,\s]+\]', '', response)
        # Clean up empty bold/italic markers left behind
        response = re.sub(r'\*{2,3}\s*\*{2,3}', '', response).strip()

        # Strip fenced code blocks (```...```) — replace with their inner content
        response = re.sub(r'```[^\n]*\n(.*?)```', lambda m: m.group(1).strip(), response, flags=re.DOTALL)
        # Strip leading whitespace from lines to prevent indented code-block rendering
        response = "\n".join(line.lstrip() for line in response.splitlines()).strip()

        # Get reference file paths from structured data
        refs = [
            r["file_path"]
            for r in (result.get("data") or {}).get("references", [])
            if r.get("file_path")
        ]

        # Get retrieved chunk texts for evaluation (RAGAS contexts)
        chunks = [
            c["content"]
            for c in (result.get("data") or {}).get("chunks", [])
            if c.get("content")
        ]

        return {"query": question, "response": response, "references": refs, "chunks": chunks, "mode": effective_mode}

    def clear_history(self):
        self._history.clear()
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def finalize(self):
        self._loop.run_until_complete(self.rag.finalize_storages())
        logger.info("Storages finalized")