import logging
from functools import lru_cache
from typing import Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLaMAConfig:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct",
                 max_length: int = 150, min_length: int = 10, 
                 temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9, 
                 no_repeat_ngram_size: int = 3, max_new_tokens: int = 64,
                 device: Optional[str] = None):
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

class LLaMAModel:
    def __init__(self, config: LLaMAConfig):
        self.config = config
        self.device = config.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        model_config = AutoConfig.from_pretrained(config.model_name, is_decoder=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, config=model_config).to(self.device)
        self.model.eval()

    @lru_cache(maxsize=32)
    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        return inputs

    def generate_text(self, prompt: str, system_message: str = "You are a helpful assistant. Provide brief, direct answers, prepared and formatted for a chat interface.") -> str:
        formatted_prompt = f"{system_message}\n\n{prompt}"
        logger.info(f"Model input:\n\n{formatted_prompt}\n\n")
        inputs = self._prepare_inputs(formatted_prompt)
        
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            do_sample=True if self.config.temperature > 0 else False,
            num_beams=3,
            pad_token_id=self.tokenizer.eos_token_id,
            min_length=self.config.min_length,
            max_length=self.config.max_length,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            early_stopping=True
        )
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)
        
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        response = decoded[len(formatted_prompt):].strip()
        
        if "User:" in response:
            response = response.split("User:")[0].strip()
        
        return response.strip()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def save(self, path: str) -> None:
        try:
            torch.save(self.state_dict(), path)
            logger.info(f"Model saved at {path}")
        except Exception as e:
            logger.error("Error saving model", exc_info=e)

    def load(self, path: str) -> None:
        try:
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info(f"Loaded model state from {path}")
        except Exception as e:
            logger.error("Error loading model state", exc_info=e)