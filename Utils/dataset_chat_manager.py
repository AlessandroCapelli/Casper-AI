"""
Dataset Chat Manager

This module handles the processing and cleaning of chat messages from a dataset. It provides
functionality to validate and clean chat messages based on configurable parameters like
sender names, minimum message length, and invalid message patterns.

Dependencies:
    - csv: For handling CSV file operations
    - re: For regular expression pattern matching
    - pathlib: For path handling
    - typing: For type hints
"""

import csv
import re
from pathlib import Path
from typing import List, Tuple, Optional

class DatasetChatManager:
    def __init__(self, 
                 query_sender: str = "",
                 response_sender: str = "",
                 min_message_length: int = 5,
                 invalid_messages: Optional[List[str]] = None):
        """
        Initialize the DatasetChatManager with configuration parameters.
        
        Args:
            query_sender (str): The name of the sender who asks questions
            response_sender (str): The name of the sender who provides responses
            min_message_length (int): Minimum length for valid messages
            invalid_messages (List[str], optional): List of message patterns to exclude
        """
        self.query_sender = query_sender
        self.response_sender = response_sender
        self.min_message_length = min_message_length
        self.invalid_messages = invalid_messages or [
            'image omitted',
            'audio omitted',
            'video omitted',
            'document omitted',
            'sticker omitted',
            'Missed voice call',
            'Missed video call'
        ]

    def clean_message(self, message: str) -> str:
        # Remove special Unicode characters and formatting
        cleaned = re.sub(r'[^\x00-\x7F]+', '', message)
        # Clean up any remaining special characters
        cleaned = re.sub(r'\[U\+[0-9A-F]+\]', '', cleaned)
        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def is_valid_message(self, message: str) -> bool:
        # Skip messages that are just notifications or media
        return (
            message and 
            message.strip() and 
            not any(invalid in message for invalid in self.invalid_messages)
        )

    def extract_messages(self, lines: List[str]) -> List[Tuple[str, str]]:
        messages = []
        current_sender = None
        current_messages = []
        
        for line in lines:
            # Clean the line first
            line = self.clean_message(line)
            # Extract timestamp, sender and message
            match = re.match(r"\[\d{2}/\d{2}/\d{2}, \d{2}:\d{2}:\d{2}\] (.*?): (.*)", line)
            if match:
                sender = match.group(1)
                message = match.group(2)
                
                # Only process valid messages
                if self.is_valid_message(message):
                    # If this is a new sender or first message
                    if current_sender is None:
                        current_sender = sender
                        current_messages = [message]
                    # If same sender, append to current messages
                    elif sender == current_sender:
                        current_messages.append(message)
                    # If different sender, save previous messages and start new
                    else:
                        if current_messages:  # Save previous message group
                            combined_message = " ".join(current_messages)
                            messages.append((current_sender, combined_message))
                        current_sender = sender
                        current_messages = [message]
        
        # Don't forget to add the last group of messages
        if current_messages:
            combined_message = " ".join(current_messages)
            messages.append((current_sender, combined_message))
        
        return messages

    def create_dataset(self, input_file: str, output_file: str) -> int:
        """
        Create a dataset from a chat file and save it to a CSV file.
        
        Args:
            input_file (str): Path to the input chat file
            output_file (str): Path to save the output CSV file
            
        Returns:
            int: Number of conversation pairs created
        """
        # Ensure input file exists
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Read the chat file
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Extract and clean messages
        messages = self.extract_messages(lines)

        # Create query-response pairs
        query_response_pairs = []
        for i in range(len(messages) - 1):
            sender, message = messages[i]
            next_sender, next_message = messages[i + 1]
            
            # Only include conversations where query_sender asks and response_sender responds
            if sender == self.query_sender and next_sender == self.response_sender:
                # Additional validation for meaningful conversations
                if len(message) > self.min_message_length and len(next_message) > self.min_message_length:
                    query_response_pairs.append((message, next_message))

        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['query', 'response'])
            writer.writerows(query_response_pairs)
            
        print(f"Dataset created with {len(query_response_pairs)} conversation pairs")
        print(f"Messages were combined when sent consecutively by the same person")
        print(f"Query sender: {self.query_sender}")
        print(f"Response sender: {self.response_sender}")
        
        return len(query_response_pairs) 