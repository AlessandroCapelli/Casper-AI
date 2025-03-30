"""
Experimental Lab for LLM Fine-Tuning and Evaluation
====================================================

This module provides a generic framework for:
- Loading and tokenizing datasets.
- Fine-tuning language models using Parameter Efficient Fine-Tuning (PEFT) techniques (e.g., LoRA).
- Evaluating model performance.
- Making predictions with the trained model.

Reference: Hugging Face Transformers Documentation - https://huggingface.co/docs/transformers/
"""

import json
import logging
from functools import lru_cache
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, PeftModel
from datasets import Dataset

# Configure logging to display info level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_config(file_path: str) -> dict:
    """Load the JSON configuration file."""
    with open(file_path, "r") as f:
        config = json.load(f)
    return config


class LLMConfig:
    """
    Generic configuration class for an LLM experimental lab.

    This class holds parameters for model initialization and text generation.
    It enables customization of the base model, generation limits, sampling parameters,
    and device settings.
    """
    def __init__(self, config: dict):
        self.model_name = config.get("model_name")
        self.system_message = config.get("system_message")
        self.max_length = config.get("max_length")
        self.min_length = config.get("min_length")
        self.temperature = config.get("temperature")
        self.top_k = config.get("top_k")
        self.top_p = config.get("top_p")
        self.no_repeat_ngram_size = config.get("no_repeat_ngram_size")
        self.max_new_tokens = config.get("max_new_tokens")
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = config.get("model_path")
        self.finetune = config.get("finetune_config")

        self.finetune_epochs = self.finetune.get("epochs")
        self.finetune_batch_size = self.finetune.get("batch_size")
        self.finetune_learning_rate = self.finetune.get("learning_rate")
        self.finetune_weight_decay = self.finetune.get("weight_decay")
        self.finetune_eval_strategy = self.finetune.get("eval_strategy")
        self.finetune_save_strategy = self.finetune.get("save_strategy")
        self.finetune_logging_steps = self.finetune.get("logging_steps")
        self.finetune_lora = self.finetune.get("lora_config")

        self.finetune_lora_config = LoraConfig(
            task_type = self.finetune_lora.get("task_type"),
            r = self.finetune_lora.get("r"),
            lora_alpha = self.finetune_lora.get("lora_alpha"),
            lora_dropout = self.finetune_lora.get("lora_dropout"),
            target_modules = self.finetune_lora.get("target_modules")
        )

class LLMModel:
    """
    Generic LLM Model class for an experimental lab.

    This class manages the initialization of a base model, fine-tuning on a training dataset,
    text generation for chat-style interactions, dataset preparation, and model state management.
    """
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM model with the provided configuration.

        This includes loading the tokenizer, model configuration, and the pre-trained model,
        then moving the model to the designated device and setting it to evaluation mode.

        Args:
            config (LLMConfig): An instance of LLMConfig containing model parameters.
        """
        self.config = config
        self.device = config.device

        # Load the tokenizer with a fast implementation if available
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True, add_prefix_space=True)
        self.tokenizer.truncation_side = "left"

        # Load the model configuration, assuming a decoder-only (causal) architecture
        model_config = AutoConfig.from_pretrained(config.model_name, is_decoder=True)

        # Load the base pre-trained model and move it to the specified device
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, config=model_config).to(self.device)

        # Ensure the tokenizer has a pad token; if not, add one and update the model's embeddings.
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.eval()  # Set model to evaluation mode

    @lru_cache(maxsize=32)
    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        Prepare tokenized inputs from the given prompt.

        This function tokenizes the input text and transfers the resulting tensors
        to the configured device. An LRU cache is used to avoid redundant tokenizations.

        Args:
            prompt (str): The input prompt text.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing tokenized inputs.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        return inputs

    def generate_text(self, prompt: str) -> str:
        """
        Generate text based on the provided prompt.

        Optionally prepends a system message to the prompt if not already present,
        prepares the inputs, sets up the generation configuration, and generates the output text.

        Args:
            prompt (str): The user's input prompt.

        Returns:
            str: The generated text response with the original prompt removed.
        """
        # Prepend the system message if not already in the prompt
        if self.config.system_message not in prompt:
            formatted_prompt = f"{self.config.system_message}\n{prompt}"
        else:
            formatted_prompt = prompt

        # Prepare the tokenized inputs using the cached method
        inputs = self._prepare_inputs(formatted_prompt)

        # Define the generation configuration parameters
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            do_sample=True if self.config.temperature > 0 else False,
            pad_token_id=self.tokenizer.eos_token_id,  # Use EOS token for padding
            min_length=self.config.min_length,
            max_length=inputs["input_ids"].shape[-1] + self.config.max_new_tokens,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            num_beams=3,
            early_stopping=True
        )

        # Generate text in inference mode (without gradient computation)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)

        # Decode the generated tokens and remove the prompt portion from the output
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = decoded[len(formatted_prompt):].strip()

        return response.strip()

    def tokenize_function(self, examples: dict, max_length: int = 512) -> dict:
        """
        Tokenize input examples using the pre-loaded tokenizer.
        This function extracts text from the input examples and applies tokenization with left-side truncation
        (to keep the most important information at the end), limiting sequences to the specified maximum length.

        Args:
            examples (dict): Dictionary with a key 'text' containing raw text data.
            max_length (int): Maximum sequence length after tokenization.

        Returns:
            dict: A dictionary containing tokenized outputs.
        """
        if not all(isinstance(item, str) for item in examples["text"]):
            raise ValueError("The 'text' field must be a list of strings.")
    
        tokenized_inputs = self.tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=max_length
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    def finetune(
        self,
        train_dataset,
        val_dataset
    ):
        """
        Fine-tune the base model on a provided training dataset.

        This method leverages Hugging Face's Trainer API to perform fine-tuning.
        After training, the fine-tuned model and tokenizer are saved to the specified output directory.
        """
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            return

        # Tokenize the training and validation datasets
        if "input_ids" not in train_dataset.features:
            train_dataset = train_dataset.map(self.tokenize_function, batched=True, remove_columns=["text"])

        if "input_ids" not in val_dataset.features:
            val_dataset = val_dataset.map(self.tokenize_function, batched=True, remove_columns=["text"])

        self.model = get_peft_model(self.model, self.config.finetune_lora_config)
        self.model.print_trainable_parameters()

        # Define training arguments for fine-tuning
        training_args = TrainingArguments(
            output_dir=self.config.model_path,
            num_train_epochs=self.config.finetune_epochs,
            per_device_train_batch_size=self.config.finetune_batch_size,
            learning_rate=self.config.finetune_learning_rate,
            weight_decay=self.config.finetune_weight_decay,
            eval_strategy=self.config.finetune_eval_strategy,
            save_strategy=self.config.finetune_save_strategy,
            load_best_model_at_end=True,
            logging_steps=self.config.finetune_logging_steps,
            logging_dir="./Logs",
            remove_unused_columns=False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )

        # Execute the fine-tuning process
        trainer.train()

        self.save(self.config.model_path, finetuned=True)

    def save(self, path: str, finetuned: bool = False) -> None:
        """
        Save the complete model state (model, tokenizer, and configuration) to a directory.
        
        Args:
            path (str): Directory path where the model and tokenizer will be saved.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}: {self.config.model_name}" + (" [FINETUNED]" if finetuned else ""))

    def load(self, path: str, finetuned: bool = False) -> None:
        """
        Load the complete model state (model, tokenizer, and configuration) from a directory.
        
        Args:
            path (str): Directory path from which the model and tokenizer will be loaded.
            finetuned (bool): Whether the model is finetuned or not.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        config = AutoConfig.from_pretrained(path)
        
        if finetuned:
            base_model = AutoModelForCausalLM.from_pretrained(self.config.model_name, config=config).to(self.device)
            self.model = PeftModel.from_pretrained(base_model, path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, config=config).to(self.device)
            
        self.tokenizer.truncation_side = "left"
        self.model.eval()
        logger.info(f"Model loaded from {path}: {self.config.model_name}" + (" [FINETUNED]" if finetuned else ""))

if __name__ == "__main__":
    config = load_json_config("config.json").get("llm_config")
    model = LLMModel(LLMConfig(config))

    train_dataset = Dataset.from_list([{"text": text} for text in ["Example"]])
    val_dataset = Dataset.from_list([{"text": text} for text in ["Example"]])

    model.finetune(train_dataset, val_dataset)
    model.load(config.get("model_path"), finetuned=True)