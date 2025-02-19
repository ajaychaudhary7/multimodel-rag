# -*- coding: utf-8 -*-
import os
import io
import gc
import fitz
import torch
import logging
import chromadb
import pytesseract
import numpy as np
from PIL import Image
from typing import Generator
from gme_inference import GmeQwen2VL
import uuid

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
PDF_FOLDER = "/home/ajay/Desktop/dev/code/multimodal_rag/papers"
DB_PATH = "vector_db"
MODEL_NAME = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_CHUNK_SIZE = 512  # Characters per text chunk
OVERLAP_SIZE = 128      # Chunk overlap characters

# Initialize ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="multimodal_rag",
    metadata={"hnsw:space": "cosine"}  # Remove metadata_config
)


class GmeEmbedder:
    def __init__(self):
        self.model = GmeQwen2VL(MODEL_NAME, device=DEVICE)
        self.text_instruction = "Represent this document chunk for retrieval:"
        self.image_instruction = "Represent this visual content for retrieval:"

    def generate_chunk_embeddings(self, text: str, images: list) -> np.ndarray:
        """Generate combined embeddings for a content chunk"""
        try:
            # Batch process text and images
            text_emb = self.model.get_text_embeddings(
                texts=[text],
                instruction=self.text_instruction
            )[0].cpu().numpy()

            if not images:
                return text_emb

            # Process images in batches
            image_embs = []
            for i in range(0, len(images), 1):  # Batch size 4 for VRAM efficiency
                batch = images[i:i+1]
                batch_emb = self.model.get_image_embeddings(
                    images=batch,
                    instruction=self.image_instruction
                ).cpu().numpy()
                image_embs.append(batch_emb)

            # Combine modalities
            image_emb = np.concatenate(image_embs).mean(axis=0)
            return 0.7 * text_emb + 0.3 * image_emb
            
        except Exception as e:
            logging.error(f"Embedding generation failed: {str(e)}")
            return None

class PDFProcessor:
    def __init__(self):
        self.min_image_size = 224
        
    def stream_chunks(self, pdf_path: str) -> Generator[dict, None, None]:
        """Generate content chunks with smart text splitting"""
        doc_id = os.path.basename(pdf_path).rsplit('.', 1)[0]
        
        with fitz.open(pdf_path) as doc:
            full_text = ""
            page_images = []
            
            for page_num, page in enumerate(doc):
                # Extract page content
                page_text = page.get_text()
                images = self._extract_images(page)
                
                # Accumulate content
                full_text += page_text + "\n"
                page_images.extend(images)
                
                # Split into chunks when exceeding size or at page boundaries
                while len(full_text) >= TEXT_CHUNK_SIZE:
                    chunk_text, full_text = self._split_text(full_text)
                    yield self._create_chunk(doc_id, chunk_text, page_images[:2])  # Keep 2 images per chunk
                    page_images = page_images[2:]  # Carry over remaining images

            # Process remaining content
            if full_text.strip():
                yield self._create_chunk(doc_id, full_text, page_images)

    def _create_chunk(self, doc_id: str, text: str, images: list) -> dict:
        """Create chunk with Chroma-compatible metadata"""
        image_paths = [self._save_image(img['raw_data']) for img in images]
        return {
            "chunk_id": f"{doc_id}_chunk_{hash(text) & 0xFFFF}",
            "text": text,
            "images": images,
            "ocr": [self._process_ocr(img['image']) for img in images],
            "metadata": {
                # Store as comma-separated string
                "image_paths": ",".join(image_paths) if image_paths else "none"
            }
        }

    def _save_image(self, image_data: bytes) -> str:
        """Save image to storage and return path"""
        img_dir = os.path.join(DB_PATH, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"{str(uuid.uuid4())}.jpg"
        path = os.path.join(img_dir, filename)
        
        # Write raw bytes to file
        with open(path, "wb") as f:
            f.write(image_data)
        
        return filename  # Return relative path

    def _split_text(self, text: str) -> tuple:
        """Split text with overlap and natural breaks"""
        split_pos = TEXT_CHUNK_SIZE - OVERLAP_SIZE
        while split_pos > 0 and text[split_pos] not in ('\n', '.', ' '):
            split_pos -= 1
        return text[:split_pos+1], text[split_pos+1 - OVERLAP_SIZE:]

    def _extract_images(self, page) -> list:
        """Extract and preprocess images"""
        images = []
        for img_info in page.get_images(full=True):
            try:
                base_image = page.parent.extract_image(img_info[0])
                img = Image.open(io.BytesIO(base_image["image"]))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = self._resize_image(img)
                images.append({"image": img, "raw_data": base_image["image"]})
            except Exception as e:
                logging.warning(f"Skipped image: {str(e)}")
        return images

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Smart image resizing"""
        w, h = img.size
        if min(w, h) < self.min_image_size:
            scale = self.min_image_size / min(w, h)
            return img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        return img

    def _process_ocr(self, image: Image.Image) -> str:
        """Safe OCR processing"""
        try:
            return pytesseract.image_to_string(image, config='--psm 3')
        except Exception as e:
            logging.error(f"OCR failed: {str(e)}")
            return ""

# Modified VectorDBManager class
class VectorDBManager:
    def __init__(self):
        self.embedder = GmeEmbedder()
        self.batch_size = 1
        
    def process_document(self, pdf_path: str):
        processor = PDFProcessor()
        batch = []
        
        for chunk in processor.stream_chunks(pdf_path):
            try:
                combined_text = f"{chunk['text']}\nOCR Results:\n" + "\n".join(chunk['ocr'])
                
                emb = self.embedder.generate_chunk_embeddings(
                    text=combined_text,
                    images=[img['image'] for img in chunk['images']]
                )
                
                if emb is not None:
                    batch.append({
                        "id": chunk['chunk_id'],
                        "text": combined_text,
                        "embedding": emb,
                        "metadata": chunk['metadata']  # Add this line
                    })

                if len(batch) >= self.batch_size:
                    self._upsert_batch(batch)
                    batch = []
                    
            except Exception as e:
                logging.error(f"Chunk processing failed: {str(e)}")
            finally:
                del chunk
                gc.collect()
                
        if batch:
            self._upsert_batch(batch)

    def _upsert_batch(self, batch: list):
        """ChromaDB-compatible upsert"""
        try:
            collection.upsert(
                ids=[item['id'] for item in batch],
                documents=[item['text'] for item in batch],
                embeddings=[item['embedding'].tolist() for item in batch],
                metadatas=[item['metadata'] for item in batch]
            )
        except Exception as e:
            logging.error(f"Batch upsert failed: {str(e)}")
        finally:
            del batch
            torch.cuda.empty_cache()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Process documents one at a time
    db_manager = VectorDBManager()
    
    for pdf_file in os.listdir(PDF_FOLDER):
        if not pdf_file.endswith(".pdf"):
            continue
            
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        logging.info(f"Processing {pdf_file}...")
        db_manager.process_document(pdf_path)
        logging.info(f"Completed {pdf_file}")