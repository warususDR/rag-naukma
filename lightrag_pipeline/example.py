import logging
from lightrag_model import LightRAGModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    
    rag = LightRAGModel(
        embedding_model="bge-m3",
        llm_model_name="mamaylum-4b",
        chroma_directory="../chroma_db",
        chroma_collection="naukma_documents_no_chunks",
    )
    
    # rag.load_from_chroma(max_documents=10)
    rag.insert_from_text_dir("./texts")
    
    questions = [
        "Яка посада у Ольги Полюхович в НаУКМА?",
        "НаУКМА, що ти знаєш про цей університет?"
        "Розкажи коротко про історію університету НаУКМА",
        "Які є вимоги для вступу на магістратуру НаУКМА?",
        "Хто такий Сергій Квіт, яка його роль в НаУКМА?"
    ]
    
    modes = ["hybrid"]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Питання: {question}")
        print(f"{'='*60}")
        
        for mode in modes:
            print(f"\nРежим: {mode} ---")
            
            result = rag.query(
                question,
                mode=mode,
            )
            
            print(f"Відповідь: {result['response']}")
        
        rag.clear_history()


if __name__ == "__main__":
    main()
