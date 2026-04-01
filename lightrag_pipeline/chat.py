import logging
import argparse
from lightrag_pipeline import LightRAGModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='LightRAG Chat Interface')
    parser.add_argument(
        '--chroma-dir',
        type=str,
        default='./chroma_db',
        help='Directory of existing Chroma DB'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='naukma_documents_no_chunks',
        help='Name of Chroma collection'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='hybrid',
        choices=['naive', 'local', 'global', 'hybrid'],
        help='Query mode for LightRAG'
    )
    parser.add_argument(
        '--load-from-chroma',
        action='store_true',
        help='Load documents from Chroma DB into LightRAG on startup'
    )
    parser.add_argument(
        '--max-docs',
        type=int,
        default=None,
        help='Maximum documents to load from Chroma (default: all)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='LLM sampling temperature'
    )
    
    args = parser.parse_args()
    
    logger.info("Initializing LightRAG model...")
    rag = LightRAGModel(
        embedding_model="bge-m3",
        llm_model_name="lapa-v0.1.2",
        chroma_directory=args.chroma_dir,
        chroma_collection=args.collection,
        temperature=args.temperature
    )
    
    if args.load_from_chroma:
        logger.info("Loading documents from Chroma DB...")
        rag.load_from_chroma(max_documents=args.max_docs)
    
    print("\n" + "="*60)
    print("НАУКМА Асистент (LightRAG + MamayLM)")
    print("="*60)
    print(f"Режим запитів: {args.mode}")
    print("\nКоманди:")
    print("  'exit' - вийти")
    print("  'clear' - очистити історію розмови")
    print("  'mode <режим>' - змінити режим (naive/local/global/hybrid)")
    print("="*60 + "\n")
    
    current_mode = args.mode
    
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
            
            if query.lower().startswith('mode '):
                new_mode = query[5:].strip().lower()
                if new_mode in ['naive', 'local', 'global', 'hybrid']:
                    current_mode = new_mode
                    print(f"Режим змінено на: {current_mode}")
                else:
                    print("Невірний режим. Доступні: naive, local, global, hybrid")
                continue
            
            result = rag.query(query, mode=current_mode, use_history=True)
            
            print(f"\nАсистент: {result['response']}")
            
        except KeyboardInterrupt:
            print("\n\nДо побачення!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\nПомилка: {e}")


if __name__ == "__main__":
    main()
