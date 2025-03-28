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

import logging
from functools import lru_cache
from typing import Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig
from datasets import Dataset

# Configure logging to display info level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMConfig:
    """
    Generic configuration class for an LLM experimental lab.

    This class holds parameters for model initialization and text generation.
    It enables customization of the base model, generation limits, sampling parameters,
    and device settings.
    """
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct",
        system_message: str = "You are a helpful assistant. Provide brief, direct answers in a chat interface.",
        max_length: int = 128,
        min_length: int = 8,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        no_repeat_ngram_size: int = 3,
        max_new_tokens: int = 64,
        device: Optional[str] = None,
    ):
        """
        Initialize LLM configuration parameters.

        Args:
            model_name (str): Identifier for the base pre-trained model.
            system_message (str): A guiding system message for model behavior.
            max_length (int): Maximum total length (prompt + generated tokens).
            min_length (int): Minimum length of the generated sequence.
            temperature (float): Sampling temperature to control randomness.
            top_k (int): Number of highest probability tokens to consider for top-k sampling.
            top_p (float): Cumulative probability threshold for nucleus sampling.
            no_repeat_ngram_size (int): Prevents the repetition of n-grams of this size.
            max_new_tokens (int): Maximum number of tokens to generate.
            device (Optional[str]): Device to run the model on ('cuda' or 'cpu').
                                    If None, it automatically selects based on availability.
        """
        self.model_name = model_name
        self.system_message = system_message
        self.max_length = max_length
        self.min_length = min_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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

        # Log the input that will be fed into the model
        # logger.info(f"\n--------- MODEL INPUT START ---------\n{formatted_prompt}\n---------- MODEL INPUT END ----------\n")

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
            max_length=self.config.max_length,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
			num_beams=3,
			early_stopping=True,
        )

        # Generate text in inference mode (without gradient computation)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)

        # Decode the generated tokens and remove the prompt portion from the output
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = decoded[len(formatted_prompt):].strip()

        return response.strip()

    def finetune(
        self,
        train_dataset,
		val_dataset,
        output_dir: str = "./model",
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        eval_strategy: str = "epoch",
        save_strategy: str = "epoch",
        logging_steps: int = 500,
    ):
        """
        Fine-tune the base model on a provided training dataset.

        This method leverages Hugging Face's Trainer API to perform fine-tuning.
        After training, the fine-tuned model and tokenizer are saved to the specified output directory.

        Args:
            train_dataset: A dataset object (e.g., a Hugging Face Dataset or PyTorch Dataset)
                           containing the training examples.
			val_dataset: A dataset object (e.g., a Hugging Face Dataset or PyTorch Dataset)
                           containing the validation examples.
            output_dir (str): Directory path where the fine-tuned model and logs will be saved.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size per device.
            learning_rate (float): Learning rate for training.
            weight_decay (float): Weight decay to apply during training.
            eval_strategy (str): Evaluation strategy to use (e.g., "epoch", "steps").
            save_strategy (str): Save strategy for checkpointing (e.g., "epoch", "steps").
            logging_steps (int): Number of steps between logging outputs.
        """
        # Tokenize the dataset if not already tokenized
        if "input_ids" not in train_dataset.features:
            train_dataset = train_dataset.map(self.tokenize_function, batched=True, remove_columns=["text"])

        if "input_ids" not in val_dataset.features:
            val_dataset = val_dataset.map(self.tokenize_function, batched=True, remove_columns=["text"])

		# Print the model architecture
        # for name, module in self.model.named_modules():
        #     print(name)

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=["q_proj"]
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Define training arguments for fine-tuning
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=True,
            logging_steps=logging_steps,
            logging_dir="./Logs",
            remove_unused_columns=False,
            report_to="none",  # Disable reporting to avoid excessive output
        )

        # Initialize the Trainer for fine-tuning
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
			eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )

        # Execute the fine-tuning process
        trainer.train()

        # Save the fine-tuned model and tokenizer to the specified directory
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

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
        # Tokenize the text without returning tensors (let the data collator handle that later)
        if not all(isinstance(item, str) for item in examples["text"]):
            raise ValueError("The 'text' field must be a list of strings.")
    
        tokenized_inputs = self.tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()

        return tokenized_inputs

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Retrieve the current state dictionary of the model.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the model's parameters.
        """
        return self.model.state_dict()

    def save(self, path: str) -> None:
        """
        Save the current model state to a file.

        Args:
            path (str): The file path where the model state will be saved.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load a model state from a file.

        Args:
            path (str): The file path from which the model state will be loaded.
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

if __name__ == "__main__":
	model = LLMModel(LLMConfig())

	# Fine-tune the model
	train_dataset = Dataset.from_list([{"text": text} for text in ['Ciao a tutti']])
	val_dataset = Dataset.from_list([{"text": text} for text in ['How are you?']])

	if (len(train_dataset) > 0 and len(val_dataset) > 0):
		model.finetune(train_dataset, val_dataset, epochs=1)

	for prompt in val_dataset:
		print("Input: ", prompt["text"], "\nPrediction:", model.generate_text(prompt["text"]))