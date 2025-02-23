import logging
import os
import datetime
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from chromadb import PersistentClient
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from gme_inference import GmeQwen2VL  # your local module

# Configuration
DB_PATH = "vector_db"
GME_MODEL_NAME = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hard limit to avoid monstrous prompts (e.g., 131,072 is the model's max, but let's keep a safety margin).
MAX_TOKENS_FOR_PROMPT = 8192

class RAGSystem:
    def __init__(self):
        # Initialize the retriever using GmeQwen2VL
        self.retriever = GmeQwen2VL(GME_MODEL_NAME, device='cpu')
        self.generator = self._init_generator()
        self.collection = PersistentClient(path=DB_PATH).get_collection("multimodal_rag")

    def _init_generator(self):
        # Offload layers to CPU automatically if GPU runs out of memory, 
        # and set a large CPU max_memory if you have enough RAM (e.g. 32GB).
        max_memory_config = {
            0: "8GiB",     # GPU 0 can use ~14GiB
            "cpu": "32GiB"  # Up to 32GiB of CPU RAM
        }
        processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",         # Automatic GPU/CPU placement
            max_memory  = max_memory_config
        ).eval()
        return {"model": model, "processor": processor}

    def _retrieve_context(self, query_embedding: List[float], top_k: int = 1) -> Dict:
        # Include embeddings in the results for re-ranking.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings"]
        )
        if not results or not results.get("documents") or len(results["documents"][0]) == 0:
            logging.warning("No relevant documents found for the query.")
            return {"texts": [], "metadata": []}

        # Re-rank results using cosine similarity.
        query_emb = np.array(query_embedding)
        similarities = []
        for emb in results["embeddings"][0]:
            emb_arr = np.array(emb)
            sim = np.dot(query_emb, emb_arr) / (np.linalg.norm(query_emb) * np.linalg.norm(emb_arr) + 1e-8)
            similarities.append(sim)
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

        texts = [results["documents"][0][i] for i in sorted_indices]
        metadata = [results["metadatas"][0][i] for i in sorted_indices]
        return {"texts": texts, "metadata": metadata}

    def _create_context_content(self, pdf_context: Dict) -> List[Dict]:
        """
        Generate content items (mostly text, plus images) from the re-ranked docs.
        """
        content = []
        for text, meta in zip(pdf_context["texts"], pdf_context["metadata"]):
            meta_info = []
            if meta.get("document_id"):
                meta_info.append(f"Document: {meta['document_id']}")
            if meta.get("page") is not None:
                meta_info.append(f"Page: {meta['page']}")
            if meta.get("processed_time"):
                meta_info.append(f"Processed at: {meta['processed_time']}")
            meta_str = " | ".join(meta_info)
            enriched_text = f"{meta_str}\n{text}" if meta_str else text

            content.append({"type": "text", "text": f"\nDocument text:\n{enriched_text}"})
            
            # If there's a full_page image, it’s appended here, but not carried over in chat history.
            if meta.get("full_image_path") and meta["full_image_path"] != "none":
                for img_path in meta["full_image_path"].split(","):
                    full_path = os.path.join(DB_PATH, "images", img_path)
                    try:
                        img = Image.open(full_path)
                        content.append({"type": "image", "image": img})
                    except Exception as e:
                        logging.warning(f"Couldn't load image {img_path}: {e}")
        return content

    def _extract_citations(self, pdf_context: Dict) -> str:
        """
        Extract references from metadata (doc ID, page, time, etc.).
        """
        citations = []
        for meta in pdf_context.get("metadata", []):
            citation = []
            if meta.get("document_id"):
                citation.append(f"Document: {meta['document_id']}")
            if meta.get("page") is not None:
                citation.append(f"Page: {meta['page']}")
            if meta.get("processed_time"):
                citation.append(f"Processed at: {meta['processed_time']}")
            if meta.get("full_image_path") and meta["full_image_path"] != "none":
                citation.append(f"Image(s): {meta['full_image_path']}")
            if citation:
                citations.append(" | ".join(citation))
        return "\n".join(citations)

    def _format_prompt(
        self,
        query: str,
        context: Dict,
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Build a list of messages that includes:
        - System instruction
        - Up to last 5 text-only turns
        - The retrieved context (text)
        - Current user query
        Then convert to a prompt with apply_chat_template.
        """
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. You have short-term memory of the conversation. Use text from retrieved documents only if relevant."
                    }
                ]
            }
        ]
        # Keep last 5 turns (text only).
        if chat_history:
            for user_msg, assistant_msg in chat_history[-2:]:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": user_msg}]
                })
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_msg}]
                })
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Relevant documents (text only):"},
                *self._create_context_content(context),
                {"type": "text", "text": f"\nQuestion: {query}"}
            ]
        })

        # Turn messages into a single string (with the QWen processor).
        prompt = self.generator["processor"].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # If the final prompt is huge, we can truncate it to avoid indexing errors.
        prompt = self._truncate_prompt_if_needed(prompt)

        return prompt, messages

    def _truncate_prompt_if_needed(self, prompt: str) -> str:
        """
        If the final text would exceed a certain token threshold, we truncate it.
        Alternatively, you could do a smart summarization.
        """
        tokenizer = self.generator["processor"].tokenizer
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        length = tokens.input_ids.shape[1]
        if length > MAX_TOKENS_FOR_PROMPT:
            logging.warning(
                f"Prompt has {length} tokens, exceeding limit {MAX_TOKENS_FOR_PROMPT}. Truncating."
            )
            # Truncate to max allowed
            truncated_ids = tokens.input_ids[0, :MAX_TOKENS_FOR_PROMPT]
            prompt = tokenizer.decode(truncated_ids, skip_special_tokens=False)
        return prompt

    def generate_answer(
        self,
        query: str,
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[str, List[Image.Image], str]:
        """
        Returns: (answer_text, list_of_images, citation_string)
        """
        try:
            with torch.no_grad():
                query_embedding = self.retriever.get_text_embeddings(
                    texts=[query],
                    instruction="Represent this question for retrieving relevant documents:"
                )[0].cpu().tolist()

            context = self._retrieve_context(query_embedding, top_k=2)
            prompt, messages = self._format_prompt(query, context, chat_history)
            logging.info("Prompt ready. Generating response...")

            # Extract images from the *retrieved docs*, not from chat history.
            context_contents = self._create_context_content(context)
            images = [item["image"] for item in context_contents if item.get("type") == "image"]

            citations = self._extract_citations(context)

            # We do have images from the prompt, but we won't include older images from chat. 
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.generator["processor"](
                text=[prompt],
                images=image_inputs,  # Might be an empty list if we omit older images
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(DEVICE)

            outputs = self.generator["model"].generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.4,
                top_p=0.8,
                repetition_penalty=1.1,
                eos_token_id=self.generator["processor"].tokenizer.eos_token_id
            )
            # Remove the prompt portion from the beginning
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            output_text = self.generator["processor"].batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return output_text[0], images, citations

        except Exception as e:
            logging.error(f"Error during answer generation: {e}")
            return "Sorry, an error occurred while generating the answer.", [], ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag = RAGSystem()
    history = []
    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")
        if question.lower() in ['exit', 'quit']:
            break
        answer, imgs, citations = rag.generate_answer(question, chat_history=history)
        print(f"\nAnswer: {answer}")
        if citations:
            print(f"\nCitations:\n{citations}")
        history.append((question, answer))
