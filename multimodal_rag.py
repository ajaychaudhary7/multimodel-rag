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
        self.text_instruction = "Represent this document page for retrieval:"
        self.image_instruction = "Represent this visual content for retrieval:"
    
    def generate_page_embedding(self, text: str, images: list) -> np.ndarray:
        try:
            with torch.no_grad():
                # Optionally, convert model to FP16 if supported
                # self.model = self.model.half()  # Uncomment if your model supports FP16
                text_emb = self.model.get_text_embeddings(
                    texts=[text],
                    instruction=self.text_instruction
                )[0].cpu().numpy()
                text_emb = l2_normalize(text_emb)
                
                if not images:
                    return text_emb

                image_embs = []
                # Process images one at a time to limit peak memory usage
                for i in range(0, len(images), 1):
                    batch = images[i:i+1]
                    batch_emb = self.model.get_image_embeddings(
                        images=batch,
                        instruction=self.image_instruction
                    ).cpu().numpy()
                    image_embs.append(batch_emb)
                    
                image_emb = np.concatenate(image_embs).mean(axis=0)
                image_emb = l2_normalize(image_emb)
                
                combined_emb = 0.7 * text_emb + 0.3 * image_emb
                return l2_normalize(combined_emb)
                
        except Exception as e:
            logging.error(f"Embedding generation failed: {str(e)}")
            return None


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


class PDFProcessor:
    def __init__(self):
        self.min_image_size = 224
        
    def stream_chunks(self, pdf_path: str) -> Generator[dict, None, None]:
        """Process one page at a time, yielding one chunk per page."""
        doc_id = os.path.basename(pdf_path).rsplit('.', 1)[0]
        
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                # Generate full-page screenshot
                page_image = self._render_page(page)
                
                # Optionally, subdivide the page screenshot into parts (e.g., 4-6 parts)
                subdivided_images = self._subdivide_image(page_image, parts=1)
                
                # Perform OCR on full screenshot for additional verification
                ocr_text = self._process_ocr(page_image)
                
                # Create chunk metadata including page number and image info
                yield self._create_chunk(doc_id, page_text, page_image, subdivided_images, ocr_text, page_num)
    
    def _render_page(self, page) -> Image.Image:
        """Render the entire page as an image."""
        # Convert the page to an image (using a zoom factor for quality if desired)
        zoom = 2  # adjust zoom as needed for resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    def _subdivide_image(self, img: Image.Image, parts: int = 4) -> list:
        """Divide the image into several equal parts (horizontally or in a grid)."""
        width, height = img.size
        # For simplicity, let's divide the image vertically into equal segments
        part_height = height // parts
        sub_images = []
        for i in range(parts):
            box = (0, i * part_height, width, (i + 1) * part_height)
            sub_images.append(img.crop(box))
        return sub_images

    def _create_chunk(self, doc_id: str, text: str, full_image: Image.Image, subdivided_images: list, ocr_text: str, page_num: int) -> dict:
        """Create chunk with enhanced metadata including screenshots and page number."""
        # Save the full page screenshot
        full_image_path = self._save_image(self._image_to_bytes(full_image))
        # Optionally, save subdivided images
        subdivided_paths = [self._save_image(self._image_to_bytes(img)) for img in subdivided_images]
        metadata = {
            "document_id": doc_id,
            "page": page_num,
            "full_image_path": full_image_path,
            "subdivided_image_paths": ",".join(subdivided_paths) if subdivided_paths else "none"
        }
        return {
            "chunk_id": f"{doc_id}_page_{page_num}",
            "text": text,
            "images": [ {"image": full_image} ] + [{"image": img} for img in subdivided_images],
            "ocr": [ocr_text],
            "metadata": metadata
        }
    
    def _image_to_bytes(self, img: Image.Image) -> bytes:
        """Convert PIL Image to raw bytes."""
        with io.BytesIO() as output:
            img.save(output, format="JPEG")
            return output.getvalue()

    def _save_image(self, image_data: bytes) -> str:
        """Save image data to storage and return the filename."""
        img_dir = os.path.join(DB_PATH, "images")
        os.makedirs(img_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join(img_dir, filename)
        with open(path, "wb") as f:
            f.write(image_data)
        return filename  # Relative path

    def _process_ocr(self, image: Image.Image) -> str:
        try:
            return pytesseract.image_to_string(image, config='--psm 3')
        except Exception as e:
            logging.error(f"OCR failed: {str(e)}")
            return ""


# Updated VectorDBManager class with retrieval performance improvements
class VectorDBManager:
    def __init__(self):
        self.embedder = GmeEmbedder()  # Now using the updated embedding function for pages
        self.batch_size = 1  # Adjust batch size as needed for throughput
        
    def process_document(self, pdf_path: str):
        processor = PDFProcessor()
        batch = []
        
        for chunk in processor.stream_chunks(pdf_path):
            try:
                # Combine the page text with OCR results to provide rich context
                combined_text = f"{chunk['text']}\nOCR Results:\n" + "\n".join(chunk['ocr'])
                
                # Use the updated page-based embedding function
                emb = self.embedder.generate_page_embedding(
                    text=combined_text,
                    images=[img['image'] for img in chunk['images']]
                )
                
                if emb is not None:
                    # Enrich metadata with processing time (optional)
                    chunk_metadata = chunk['metadata']
                    chunk_metadata["processed_time"] =  str(datetime.datetime.now())
                    
                    batch.append({
                        "id": chunk['chunk_id'],
                        "text": combined_text,
                        "embedding": emb,
                        "metadata": chunk_metadata
                    })
                    
                # Upsert when batch is full
                if len(batch) >= self.batch_size:
                    self._upsert_batch(batch)
                    batch = []
                    
            except Exception as e:
                logging.error(f"Chunk processing failed for {chunk.get('chunk_id', 'unknown')}: {str(e)}")
            finally:
                del chunk
                gc.collect()
                
        if batch:
            self._upsert_batch(batch)

    def _upsert_batch(self, batch: list):
        """ChromaDB-compatible upsert with improved error logging and memory cleanup."""
        try:
            collection.upsert(
                ids=[item['id'] for item in batch],
                documents=[item['text'] for item in batch],
                embeddings=[item['embedding'].tolist() for item in batch],
                metadatas=[item['metadata'] for item in batch]
            )
            logging.info(f"Successfully upserted batch of {len(batch)} chunks.")
        except Exception as e:
            logging.error(f"Batch upsert failed: {str(e)}")
        finally:
            del batch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import datetime
    logging.basicConfig(level=logging.INFO)
    
    # Make sure the ChromaDB collection is configured with improved indexing parameters
    # (Ensure these parameters are supported by your ChromaDB version.)
    collection = client.get_or_create_collection(
        name="multimodal_rag",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:ef": "200",
            "hnsw:efConstruction": "200"
        }
    )
    
    # Process documents one at a time
    db_manager = VectorDBManager()
    
    for pdf_file in os.listdir(PDF_FOLDER):
        if not pdf_file.endswith(".pdf"):
            continue
            
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        logging.info(f"Processing {pdf_file}...")
        db_manager.process_document(pdf_path)
        logging.info(f"Completed {pdf_file}")
