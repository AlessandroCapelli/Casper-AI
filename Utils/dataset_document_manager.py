"""
Dataset Document Manager

This module handles the processing of a dataset using OpenAI's GPT model to generate responses
for queries. It reads from a CSV file containing queries, processes each query through the
OpenAI API, and saves the responses back to a new CSV file.

Dependencies:
    - pandas: For CSV file handling
    - tqdm: For progress bar visualization
    - openai: For OpenAI API integration
"""

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# OpenAI API configuration
api_key = "<your_api_key>"
client = OpenAI(api_key=api_key)

def inference(query):
	"""
	Process a single query through the OpenAI API to generate a response.
	
	Args:
		query (str): The input query to be processed
		
	Returns:
		str: The generated response from the OpenAI model
	"""
	response = client.chat.completions.create(
		model="gpt-4o-mini",
		messages= [
			{
				"role": "system",
				"content": [
					{
						"type": "text",
						"text": "You are a helpful assistant that can answer questions about the document."
					}
				]
			},
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": query
					}
				]
			}
		],
		response_format= {
			"type": "text"
		},
		temperature=1,
		max_completion_tokens=2048,
		top_p=1,
		frequency_penalty=0,
		presence_penalty=0
	)

	return response.choices[0].message.content

# Read the input dataset
df = pd.read_csv('dataset.csv')

# Process each query and store responses
responses = []
for query in tqdm(df['query'].values, desc="Processing queries"):
	response = inference(query)
	responses.append(response)

# Add responses to the dataframe and save to new CSV
df['response'] = responses
df.to_csv('dataset_final.csv', index=False)
