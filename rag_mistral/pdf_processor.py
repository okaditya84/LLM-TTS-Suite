import os
import logging
from typing import List, Dict
from pathlib import Path
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 - faster but less accurate"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber - more accurate but slower"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_text_hybrid(self, pdf_path: str) -> str:
        """Hybrid approach: try pdfplumber first, fallback to PyPDF2"""
        text = self.extract_text_pdfplumber(pdf_path)
        if len(text.strip()) < 100:  # If extraction seems poor
            logger.info(f"Falling back to PyPDF2 for {pdf_path}")
            text = self.extract_text_pypdf2(pdf_path)
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        # Remove page numbers and headers/footers (basic approach)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be page numbers
            if len(line) > 3 and not line.isdigit():
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    def process_single_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF file"""
        logger.info(f"Processing {pdf_path}")
        
        # Extract text
        raw_text = self.extract_text_hybrid(pdf_path)
        if not raw_text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return []
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Create document metadata
        file_name = Path(pdf_path).name
        file_hash = hashlib.md5(cleaned_text.encode()).hexdigest()[:8]
        
        # Split into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": file_name,
                    "file_path": pdf_path,
                    "chunk_id": f"{file_hash}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_pdf_directory(self, directory_path: str) -> List[Document]:
        """Process all PDFs in a directory"""
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            documents = self.process_single_pdf(str(pdf_file))
            all_documents.extend(documents)
        
        logger.info(f"Created {len(all_documents)} document chunks from {len(pdf_files)} PDFs")
        return all_documents
