import logging
from data_processing import PDFExtractor, VectorStoreBuilder


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    PDF_DIRECTORY = "./documents"
    JSON_OUTPUT = "processed_documents.json"
    CHROMA_DB_PATH = "./chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_FILES = None
    
    extractor = PDFExtractor()
    documents = extractor.extract_from_directory(
        directory=PDF_DIRECTORY,
        max_files=MAX_FILES
    )
    
    if not documents:
        logger.error("No documents were processed. Exiting.")
        return
    
    logger.info(f"\nSuccessfully extracted {len(documents)} documents")
    
    builder = VectorStoreBuilder(
        embedding_model="BAAI/bge-m3",
        collection_name="naukma_documents_no_chunks",
        persist_directory=CHROMA_DB_PATH
    )
    
    builder.save_documents_to_json(documents, JSON_OUTPUT)
    
    builder.add_documents_to_vector_store(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        batch_size=100
    )
    
    test_query = "В якому році було введено Про введення в дію Положення про етику наукових досліджень та запобігання неправомірній поведінці при проведенні досліджень?"
    results = builder.query_vector_store(test_query, n_results=3)
    
    logger.info(f"\nTop 3 results for query: '{test_query}'")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        logger.info(f"\nResult {i + 1} (distance: {distance:.4f}) ---")
        logger.info(f"Filename: {metadata['filename']}")
        logger.info(f"Chunk: {metadata['chunk_index'] + 1}/{metadata['total_chunks']}")
        logger.info(f"Text preview: {doc[:500]}...")
    
    logger.info(f"Vector store saved to: {CHROMA_DB_PATH}")
    logger.info(f"JSON documents saved to: {JSON_OUTPUT}")


if __name__ == "__main__":
    main()
