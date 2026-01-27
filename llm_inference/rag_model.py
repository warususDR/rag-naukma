import logging
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests


logger = logging.getLogger(__name__)


class RAGModel:
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        collection_name: str = "naukma_documents_no_chunks",
        persist_directory: str = "./chroma_db",
        ollama_model: str = "llama3",
        ollama_url: str = "http://localhost:11434",
        top_k: int = 3
    ):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.top_k = top_k
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        logger.info(f"Connecting to Chroma at {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_collection(name=collection_name)
        logger.info(f"Connected to collection '{collection_name}' with {self.collection.count()} documents")
        
        self.conversation_history: List[Dict[str, str]] = []
    
    def retrieve_context(self, query: str, n_results: int = None) -> List[Dict]:
        n_results = n_results or self.top_k
        
        query_embedding = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        contexts = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            contexts.append({
                'text': doc,
                'metadata': metadata,
                'distance': distance,
                'rank': i + 1
            })
        
        return contexts
    
    def generate_response(
        self,
        query: str,
        contexts: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        if system_prompt is None:
            system_prompt = """Ти - помічник для відповідей на питання про НаУКМА (Національний університет "Києво-Могилянська академія").
Використовуй надані документи для відповіді на питання користувача.
Відповідай українською мовою, будь точним та конкретним.
Якщо в документах немає інформації для відповіді, так і скажи. 
Надай ім'я документу, лише якщо користувач попросив зробити це. 
Ігноруй незрозумілі символи та неточності, бо таке може бути в наданих документах, отриманих за допомогою ocr."""
        
        context_text = "\n\n".join([
            f"Документ {ctx['rank']} (з файлу {ctx['metadata']['filename']}):\n{ctx['text']}"
            for ctx in contexts
        ])
        
        prompt = f"""Контекст з документів:
{context_text}

Питання користувача: {query}

Відповідь:"""
        
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result['response']
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def chat(
        self,
        query: str,
        use_history: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict:
        logger.info(f"Query: {query}")
        
        contexts = self.retrieve_context(query)
        
        logger.info(f"Retrieved {len(contexts)} contexts")
        for ctx in contexts:
            logger.info(f"  - {ctx['metadata']['filename']} (distance: {ctx['distance']:.4f})")
        
        if use_history and self.conversation_history:
            history_text = "\n".join([
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in self.conversation_history[-3:]
            ])
            enhanced_query = f"Попередня розмова:\n{history_text}\n\nНове питання: {query}"
        else:
            enhanced_query = query
        
        response = self.generate_response(
            enhanced_query,
            contexts,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.conversation_history.append({
            'user': query,
            'assistant': response
        })
        
        return {
            'query': query,
            'response': response,
            'contexts': contexts,
            'sources': [ctx['metadata']['filename'] for ctx in contexts]
        }
    
    def clear_history(self):
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history
