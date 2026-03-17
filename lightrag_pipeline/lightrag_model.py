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
    "Ти маєш надавати перевагу українській мові у своїх відповідях, уникай англійської мови."
)

# Simple greetings / non-informational queries that don't need RAG retrieval
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

        self.embedding_func = create_embedding_func(model_name=embedding_model, embedding_dim=1024, max_token_size=8192, host=ollama_host)

        logger.info("Initializing LightRAG...")

        self.rag = LightRAG(
            llm_model_func=self._llm_model_func,
            llm_model_name=self._llm_model_name,
            llm_model_kwargs={
                "options": {
                    "temperature": self._temperature,
                    "repeat_penalty": 1.3,
                    "repeat_last_n": 256,
                    "num_ctx": 30000,
                },
            },
            embedding_func=self.embedding_func,
            summary_max_tokens=1200,
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

        # Conversation history in LightRAG-native format
        self._history: List[Dict[str, str]] = []

        logger.info("LightRAG model initialized successfully")

    # ------------------------------------------------------------------
    # LLM wrapper (called by LightRAG internally for ALL llm calls)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------
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

    def load_from_chroma(self, max_documents: Optional[int] = None):
        if self.chroma_collection is None:
            logger.warning("No Chroma collection available")
            return

        count = self.chroma_collection.count()
        limit = min(count, max_documents) if max_documents else count

        logger.info(f"Loading {limit} documents from Chroma DB...")

        batch_size = 100
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

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, question: str, mode: str = "hybrid") -> Dict:
        logger.info(f"Query: {question} (mode: {mode})")

        # Short-circuit simple greetings — skip heavy retrieval
        if _GREETING_PATTERN.match(question.strip()):
            logger.info("Greeting detected — using bypass mode")
            effective_mode = "bypass"
        else:
            effective_mode = mode

        param = QueryParam(
            mode=effective_mode,
            top_k=40,
            conversation_history=list(self._history),
            user_prompt=NAUKMA_SYSTEM_PROMPT,
        )

        response = self._loop.run_until_complete(
            self.rag.aquery(query=question, param=param)
        )

        # Append to native history
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": response})

        # Keep last 6 messages (3 turns)
        if len(self._history) > 6:
            self._history = self._history[-6:]

        return {"query": question, "response": response, "mode": effective_mode}

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def clear_history(self):
        self._history.clear()
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def finalize(self):
        """Finalize storages for clean shutdown."""
        self._loop.run_until_complete(self.rag.finalize_storages())
        logger.info("Storages finalized")