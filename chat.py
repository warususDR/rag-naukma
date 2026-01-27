import logging
from llm_inference import RAGModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    rag = RAGModel(
        embedding_model="BAAI/bge-m3",
        collection_name="naukma_documents_no_chunks",
        persist_directory="./chroma_db",
        ollama_model="llama3:8b",
        top_k=3
    )
    
    print("НАУКМА Асистент")
    print("Команди: 'exit' - вийти, 'clear' - очистити історію")
    
    while True:
        try:
            query = input("\nВи: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                print("До побачення!")
                break
            
            if query.lower() == 'clear':
                rag.clear_history()
                print("Історію розмови очищено.")
                continue
            
            result = rag.chat(query, use_history=True)
            
            print(f"\nАсистент: {result['response']}")
            print(f"\nДжерела: {', '.join(result['sources'])}")
            
        except KeyboardInterrupt:
            print("\n\nДо побачення!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nПомилка: {e}")


if __name__ == "__main__":
    main()
