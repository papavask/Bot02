# pdf_indexer.py
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import os
import argparse
from pathlib import Path

class SimplePDFIndexer:
    def __init__(self, db_path="./chroma_db"):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("pdf_documents")
    
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF file"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Simple text chunking"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks
    
    def index_pdf(self, pdf_path):
        """Process and index a PDF file"""
        print(f"Processing: {pdf_path}")
        
        # Extract text
        text = self.extract_pdf_text(pdf_path)
        if not text.strip():
            print(f"Warning: No text extracted from {pdf_path}")
            return False
        
        # Chunk text
        chunks = self.chunk_text(text)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.embedder.encode(chunks).tolist()
        
        # Create IDs and metadata
        doc_name = os.path.basename(pdf_path)
        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc_name, "chunk_id": i} for i in range(len(chunks))]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully indexed {pdf_path}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Index PDF files to ChromaDB')
    parser.add_argument('pdf_path', help='Path to PDF file or directory containing PDFs')
    args = parser.parse_args()
    
    indexer = SimplePDFIndexer()
    
    pdf_path = Path(args.pdf_path)
    
    if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
        indexer.index_pdf(pdf_path)
    elif pdf_path.is_dir():
        # Process all PDFs in directory
        pdf_files = list(pdf_path.glob("*.pdf"))
        for pdf_file in pdf_files:
            indexer.index_pdf(pdf_file)
    else:
        print("Please provide a valid PDF file or directory")

if __name__ == "__main__":
    main()
