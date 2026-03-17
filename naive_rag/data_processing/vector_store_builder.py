import json
from typing import List
import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from .pdf_preprocessor import Document


logger = logging.getLogger(__name__)


class VectorStoreBuilder:
    
    def __init__(
        self, 
        embedding_model: str = "BAAI/bge-m3",
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db"
    ):
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        logger.info(f"Initializing Chroma client at {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "naukma documents collection"}
        )
        
        logger.info(f"Collection '{collection_name}' ready. Current size: {self.collection.count()}")
    
    def save_documents_to_json(self, documents: List[Document], output_path: str = "processed_documents.json"):
        docs_data = [doc.to_dict() for doc in documents]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(documents)} documents to {output_path}")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < text_len:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= text_len:
                break
        
        return chunks
    
    def add_documents_to_vector_store(self, documents: List[Document], chunk_size: int = 1000, overlap: int = 200, batch_size: int = 100):
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        chunk_counter = 0
        
        for doc_idx, doc in enumerate(documents):
            logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {doc.filename}")
            
            chunks = self.chunk_text(doc.text, chunk_size, overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc.filename}_{chunk_idx}"
                
                all_chunks.append(chunk)
                all_metadatas.append({
                    "filename": doc.filename,
                    "filepath": doc.filepath,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "document_index": doc_idx
                })
                all_ids.append(chunk_id)
                chunk_counter += 1
        
        logger.info(f"Created {chunk_counter} chunks from {len(documents)} documents")
        
        for i in range(0, len(all_chunks), batch_size):
            batch_end = min(i + batch_size, len(all_chunks))
            batch_chunks = all_chunks[i:batch_end]
            batch_metadatas = all_metadatas[i:batch_end]
            batch_ids = all_ids[i:batch_end]
            
            logger.info(f"Encoding batch {i // batch_size + 1}/{(len(all_chunks) + batch_size - 1) // batch_size}")
            
            embeddings = self.embedding_model.encode(
                batch_chunks,
                show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()
            
            self.collection.add(
                documents=batch_chunks,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            logger.info(f"Added batch {i // batch_size + 1} ({batch_end - i} chunks)")
        
        logger.info(f"Successfully added all chunks. Total collection size: {self.collection.count()}")
    
    def query_vector_store(self, query: str, n_results: int = 5):
        logger.info(f"Querying: '{query[:50]}...'")
        
        query_embedding = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
