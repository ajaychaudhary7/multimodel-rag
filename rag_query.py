import logging
import os
import torch
from PIL import Image
from typing import List, Dict
from chromadb import PersistentClient
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration  # from HF page
from gme_inference import GmeQwen2VL  # your local module

# Configuration
DB_PATH = "vector_db"
GME_MODEL_NAME = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RAGSystem:
    def __init__(self):
        # Initialize the retriever using GmeQwen2VL.
        self.retriever = GmeQwen2VL(GME_MODEL_NAME, device=DEVICE)
        self.generator = self._init_generator()
        self.collection = PersistentClient(path=DB_PATH).get_collection("multimodal_rag")

    def _init_generator(self):
        # Load the processor and model as per Hugging Face guidelines.
        processor = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
        return {"model": model, "processor": processor}

    def _retrieve_context(self, query_embedding: List[float], top_k: int = 3) -> Dict:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        return {
            "texts": results["documents"][0],
            "metadata": results["metadatas"][0]
        }

    def _create_context_content(self, pdf_context: Dict) -> List[Dict]:
        """Generate a list of content items (text or image) from the context.
        Each content item is a dict with keys "type" and either "text" or "image". """
        content = []
        for text, meta in zip(pdf_context["texts"], pdf_context["metadata"]):
            # Add the text content.
            content.append({"type": "text", "text": f"\nDocument text:\n{text}"})
            # If images exist, attempt to load and include them.
            if meta.get("image_paths") and meta["image_paths"] != "none":
                for img_path in meta["image_paths"].split(","):
                    try:
                        img = Image.open(os.path.join(DB_PATH, "images", img_path))
                        content.append({"type": "image", "image": img})
                    except Exception as e:
                        logging.warning(f"Couldn't load image {img_path}: {e}")
        return content

    def _format_prompt(self, query: str, context: Dict) -> (str, List[Dict]):
        """
        Build a list of messages containing a system instruction,
        the retrieved context (both text and images), and the user query.
        Then use the processor's chat template to create the prompt.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer using both text and visual information from these documents:"},
                    *self._create_context_content(context),
                    {"type": "text", "text": f"\nQuestion: {query}"}
                ]
            }
        ]
        prompt = self.generator["processor"].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, messages

    def generate_answer(self, query: str) -> str:
        # 1. Retrieve context based on the query.
        query_embedding = self.retriever.get_text_embeddings(
            texts=[query],
            instruction="Represent this question for retrieving relevant documents:"
        )[0].cpu().tolist()
        context = self._retrieve_context(query_embedding)

        # 2. Build prompt and messages.
        prompt, messages = self._format_prompt(query, context)

        # 3. If there are images in the context, display them.
        # (In a notebook, you might use IPython.display.display instead.)
        context_contents = self._create_context_content(context)
        for item in context_contents:
            if item.get("type") == "image":
                try:
                    item["image"].show()
                except Exception as e:
                    logging.warning(f"Failed to display image: {e}")

        # 4. Extract multimodal inputs using a helper function.
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        # 5. Prepare model inputs.
        inputs = self.generator["processor"](
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(DEVICE)

        # 6. Generate answer.
        outputs = self.generator["model"].generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.3,
            top_p=0.8,
            repetition_penalty=1.1,
            eos_token_id=self.generator["processor"].tokenizer.eos_token_id
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        output_text = self.generator["processor"].batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag = RAGSystem()
    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")
        if question.lower() in ['exit', 'quit']:
            break
        answer = rag.generate_answer(question)
        print(f"\nAnswer: {answer}")
