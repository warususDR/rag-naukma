import logging
import asyncio
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
        temperature: float = 0.5,
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
            # llm_model_max_async=2,
            embedding_func=self.embedding_func,
            # embedding_func_max_async=4,
            summary_max_tokens=600,
            addon_params={
                "language": "Ukrainian",
                "entity_types": ["Особа", "Організація", "Місце", "Подія", "Концепція", "Документ", "Дата", "Артефакт"],
            },
            kv_storage=kv_storage,
            vector_storage=vector_storage,
            graph_storage=graph_storage,
            doc_status_storage=doc_status_storage,
        )
        
        asyncio.get_event_loop().run_until_complete(self.rag.initialize_storages())

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
        
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info("LightRAG model initialized successfully")
    
    async def _llm_model_func(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: list = [],
        **kwargs
    ) -> str:
        if system_prompt is None:
            system_prompt = """Ти - помічник для відповідей на питання про НаУКМА (Національний університет "Києво-Могилянська академія").
Використовуй надані уривки документів для відповіді на питання користувача.
Якщо в уривках виявлено граматичні чи фактичні (наприклад НаВКМА замість НаУКМА тощо) помилки та цю інформацію використано для відповіді, обов'язково виправ їх у відповіді.
Відповідай українською мовою, будь точним та конкретним.
Якщо в уривках документів немає інформації для відповіді, так і скажи.
Не надавай зайвої інформації. Не надавай неіснуючі дати, це заборонено!
Ігноруй незрозумілі символи та неточності з OCR.
Ніколи не повторюй однакові пункти у списках — кожен елемент має з'являтися лише один раз.
Перелічуй лише те, що прямо підтверджено в наданих уривках. Не вигадуй додаткових пунктів."""

        return await ollama_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            host=self._ollama_host,
            options={"temperature": self._temperature, "repeat_penalty": 1.3, "repeat_last_n": 256, "max_tokens_size": self._max_tokens},
            **kwargs
        )
    
    def insert_documents(self, texts: List[str]):
        logger.info(f"Inserting {len(texts)} documents into LightRAG...")
        self.rag.insert(texts)
        logger.info("Documents inserted successfully")

    def insert_from_text_dir(self, text_dir: str, max_files: Optional[int] = None, batch_size: int = 50):
        txt_files = sorted(Path(text_dir).glob("*.txt"))
        if max_files:
            txt_files = txt_files[:max_files]
        total = len(txt_files)
        logger.info(f"Found {total} .txt files in {text_dir}")
        for i in range(0, total, batch_size):
            batch = txt_files[i:i + batch_size]
            texts = []
            for txt_path in batch:
                try:
                    texts.append(txt_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.error(f"Could not read {txt_path.name}: {e}")
            if texts:
                self.insert_documents(texts)
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
    
    def query(
        self,
        question: str,
        mode: str = "hybrid",
        use_history: bool = False
    ) -> Dict:
        logger.info(f"Query: {question} (mode: {mode})")
        
        if use_history and self.conversation_history:
            history_text = "\n".join([
                f"Користувач: {msg['user']}\nАсистент: {msg['assistant']}"
                for msg in self.conversation_history[-3:]
            ])
            enhanced_question = f"Попередня розмова:\n{history_text}\n\nНове питання: {question}"
        else:
            enhanced_question = question

        system_prompt = """Ти - помічник для відповідей на питання про НаУКМА (Національний університет "Києво-Могилянська академія").
Використовуй надані уривки документів для відповіді на питання користувача.
Якщо в уривках виявлено граматичні чи фактичні (наприклад НаВКМА замість НаУКМА тощо) помилки та цю інформацію використано для відповіді, обов'язково виправ їх у відповіді.
Відповідай українською мовою, будь точним та конкретним.
Якщо в уривках документів немає інформації для відповіді, так і скажи.
Не надавай зайвої інформації. Не надавай неіснуючі дати, це заборонено!
Ігноруй незрозумілі символи та неточності з OCR.
Ніколи не повторюй однакові пункти у списках — кожен елемент має з'являтися лише один раз.
Перелічуй лише те, що прямо підтверджено в наданих уривках. Не вигадуй додаткових пунктів."""

        
        response = self.rag.query(
            system_prompt=system_prompt,
            query=enhanced_question,
            param=QueryParam(mode=mode)
        )

        
        self.conversation_history.append({
            'user': question,
            'assistant': response
        })
        
        return {
            'query': question,
            'response': response,
            'mode': mode
        }
    
    def clear_history(self):
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history