import torch
import os
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from huggingface_hub import HfFolder
from typing import Tuple
import config as _cfg


class T5ModelService:
    """
    Singleton T5 model service for product classification.
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __init__(self):
        """Initialize T5 service (singleton pattern)."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        print(f"🚀 Initializing T5 Service (Thread: {threading.current_thread().name})")
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        print(f"🔍 Using device: {self.device}")

        # Save HF token if provided
        if _cfg.HF_TOKEN:
            HfFolder.save_token(_cfg.HF_TOKEN)
            print(f"✅ Hugging Face token authenticated")
        else:
            print(f"⚠️ HF_TOKEN not found in environment")

        # Resolve checkpoint path
        checkpoint_path = os.path.abspath(_cfg.MODEL_PATH)
        print(f"🔍 Checkpoint path: {checkpoint_path}")
        print(f"🔍 Checkpoint exists: {os.path.exists(checkpoint_path)}")
        
        # Load tokenizer from checkpoint
        print(f"🔄 Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            print(f"✅ Tokenizer loaded from checkpoint")
        except Exception as e:
            print(f"⚠️ Tokenizer load failed: {e}")
            raise
        
        # Load base model from HF with token
        print(f"🔄 Loading base model {_cfg.BASE_MODEL_ID}...")
        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                _cfg.BASE_MODEL_ID,
                torch_dtype=dtype,
                device_map=None,
                token=_cfg.HF_TOKEN
            )
            print(f"✅ Base model loaded")
        except Exception as e:
            print(f"❌ Base model load failed: {e}")
            raise

        # Load LoRA adapters
        print(f"🔄 Loading LoRA adapters from {checkpoint_path}...")
        try:
            peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            print(f"✅ LoRA adapters loaded")
            
            # Merge
            print("🔄 Merging LoRA adapters...")
            self.model = peft_model.merge_and_unload()
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model merged and ready")
        except Exception as e:
            print(f"❌ LoRA merge failed: {e}")
            raise

        self.prefix = "Extraire nom canonique (Food/Nettoyage) :"
        self._initialized = True
        print("✅ T5 Model loaded and ready for inference!")

    @classmethod
    def get_instance(cls):
        """Double-checked locking pattern pour thread-safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("🔄 Creating T5 Service singleton instance...")
                    cls._instance = cls()
        return cls._instance

    def predict(self, description: str) -> Tuple[str, float]:
        """Generate prediction for product description."""
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not initialized. Call get_instance() first.")
        
        # Thread-safe prediction with lock to avoid concurrent conflicts
        with threading.Lock():
            input_text = f"{self.prefix}{description}"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False
                )

        # Decode
        prediction = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Confidence
        logits = torch.stack(outputs.scores, dim=1)
        probs = torch.softmax(logits, dim=-1)
        token_ids = outputs.sequences[:, 1:].unsqueeze(-1)

        if token_ids.shape[1] < probs.shape[1]:
            probs = probs[:, :token_ids.shape[1], :]

        gathered_probs = torch.gather(probs, 2, token_ids).squeeze(-1)
        confidence = gathered_probs.mean().item()
        
        return prediction, confidence