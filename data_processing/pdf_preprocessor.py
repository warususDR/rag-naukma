import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import logging
import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


logger = logging.getLogger(__name__)


@dataclass
class Document:
    text: str
    filename: str
    filepath: str
    char_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __repr__(self) -> str:
        return f"Document(filename={self.filename}, chars={self.char_count})"


class PDFExtractor:
    
    def __init__(self, langs: List[str] = None):
        self.langs = langs or ["Ukrainian", "English"]
        self.model_dict = create_model_dict(device="cuda" if torch.cuda.is_available() else "cpu")
    
    def extract_from_file(self, pdf_path: str) -> Document:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing: {os.path.basename(pdf_path)}")
        
        converter = PdfConverter(artifact_dict=self.model_dict)
        rendered = converter(pdf_path)
        result = text_from_rendered(rendered)
        full_text = result[0] if isinstance(result, tuple) else result
        
        logger.info(f"Extracted {len(full_text)} characters")
        
        doc = Document(
            text=full_text,
            filename=os.path.basename(pdf_path),
            filepath=pdf_path,
            char_count=len(full_text)
        )
        
        return doc
    
    def extract_from_directory(self, directory: str, max_files: int = None) -> List[Document]:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        pdf_files = sorted(dir_path.glob("*.pdf"))
        
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        documents = []
        
        for pdf_file in pdf_files:
            try:
                doc = self.extract_from_file(str(pdf_file))
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(documents)}/{len(pdf_files)} files")
        
        return documents
