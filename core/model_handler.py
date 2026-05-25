import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import os
import json
import re

STRUCTURED_PROMPT = (
    "You are a radiology assistant. Analyze the provided medical scan(s) and respond with "
    "ONLY a single JSON object using exactly this schema:\n"
    '{\n'
    '  "findings": "detailed observation of structures, densities, and abnormalities",\n'
    '  "impression": "concise diagnostic summary",\n'
    '  "recommendations": "suggested follow-up or next steps",\n'
    '  "confidence": "High, Moderate, or Low"\n'
    '}\n'
    "Do not include any text, markdown, or commentary outside the JSON object."
)

class MedGemmaHandler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MedGemmaHandler, cls).__new__(cls)
            cls._instance.initialized = False
            cls._instance.model_id = None
            cls._instance._hf_token = None
        return cls._instance

    def initialize(self, model_id="google/medgemma-1.5-4b-it", use_quantization=True, hf_token=None):
        # Reuse the loaded model only if both the model and token are unchanged;
        # a new token (e.g. switching HF accounts) forces a reload.
        if self.initialized and self.model_id == model_id and self._hf_token == hf_token:
            return

        self.model_id = model_id
        self._hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        try:
            # 1. Load Processor (pass the token explicitly so gated access is authenticated)
            self.processor = AutoProcessor.from_pretrained(self.model_id, token=hf_token)

            # 2. Configure Quantization if on CUDA
            bnb_config = None
            if use_quantization and self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            # 3. Load Model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                token=hf_token
            )
            
            if self.device == "cpu":
                self.model.to(self.device)

            self.initialized = True
            print(f"Model {model_id} loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.initialized = False
            raise e

    def analyze(self, images, prompt=STRUCTURED_PROMPT, max_new_tokens=1024):
        """
        Inference call for MedGemma.
        'images' can be a single PIL Image or a list of PIL Images.
        Returns the raw model string; use parse_report() to structure it.
        """
        if not self.initialized:
            return "Model not initialized."

        # Prepare messages for multimodal template
        if not isinstance(images, list):
            images = [images]

        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # On CUDA the (quantized) model computes in bfloat16, so float inputs must match.
        # On CPU the model stays float32, so only move to device without recasting.
        if self.device == "cuda":
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
        else:
            inputs = inputs.to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False # Greedy decoding as per latest recommendations
            )
            generation = generation[0][input_len:]
            
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def mock_analyze(self, scan_type="CT"):
        """Fallback for testing UI without model weights. Returns a JSON report string."""
        return json.dumps({
            "findings": (
                "No significant abnormalities detected in the immediate viewing area. "
                "Spatial alignment appears normal and tissue density indicates consistent HU values."
            ),
            "impression": f"Unremarkable {scan_type} study. No acute findings identified.",
            "recommendations": (
                "Correlation with patient history is suggested. "
                "No immediate follow-up imaging required. This is a mock analysis for UI demonstration."
            ),
            "confidence": "Moderate",
        })

    @staticmethod
    def parse_report(text):
        """Parse a model response into a structured report dict, with a safe fallback.

        Returns a dict with keys: findings, impression, recommendations, confidence.
        If the response is not valid JSON, the raw text is preserved under 'findings'.
        """
        result = {"findings": "", "impression": "", "recommendations": "", "confidence": "N/A"}
        if not text:
            return result

        # Locate the first JSON object in the response (tolerates code fences / stray text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                for key in result:
                    if data.get(key) is not None:
                        result[key] = str(data[key]).strip()
                return result
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: nothing parseable, so surface the raw text rather than losing it
        result["findings"] = text.strip()
        return result
