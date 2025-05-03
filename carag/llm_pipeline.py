""" Grounded Generation from Cache Aware Retrieval Augmented Generation Pipeline using Mistral LLM"""
# Author: Mohamed Rizwan <rizdelhi@gmail.com>

import os
import re
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from mistralai import Mistral
from typing import Optional, List, Dict
from carag.rag_pipeline import rag_pipe # import parent
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Define the GroundGeneration class using Mistral AI models.
class GroundGeneration(rag_pipe):
    def __init__(
        self,
        url:str,
        api_key: str,
        mistral_api_key: str,
        collection_name: str,
        llm_model_name: Optional[str]="mistral-large-latest",
        **kwargs
        ):
        """
        The class is designed to work with the Mistral API and Qdrant vector search engine.
        It inherits from the rag_pipe class, which handles the caching and retrieval of search results
        It uses Mistral LLMs for generating grounded responses based on search results from the Qdrant database.
        It includes methods for preparing prompts, generating responses, and parsing JSON outputs.
        The class is initialized with the Qdrant server URL, Qdrant API key, and Mistral API key.
        Get top 3 answers (grounded generation) to the user question from the top 50 re-ranked results from cache_rag_pipeline class.
        
        Args:
        url (str): The URL of the Qdrant server.
        api_key (str): The API key for the Qdrant server.
        mistral_api_key (str): The API key for the Mistral model.
        collection_name (str): The name of collection in Qdrant
        llm_model_name (str): Defaults to "mistral-large-latest"
        """
        super().__init__(url=url, api_key=api_key,collection_name=collection_name,llm_model_name=llm_model_name,mistral_api_key=mistral_api_key,**kwargs)  # Initialize the superclass
        self.collection_name=collection_name
        self.api_key=api_key or os.environ.get("api_key")
        self.mistral_api_key = mistral_api_key or os.environ.get("mistral_api_key")
        if not self.mistral_api_key:
            raise ValueError("Mistral API key is required. Please set it as an environment variable or pass it directly.")
        # Initialize the Mistral client
        self.llm_model_name =llm_model_name
        self.llm_client = Mistral(api_key=self.mistral_api_key)
        if not self.llm_client:
            raise ValueError("Failed to initialize Mistral client. Please check your API key.")
        
    def prepare_llm_prompt(self, query: str, prompt_template: str = "") -> str:
        """Prepare the prompt for the LLM.

        This method prepares the prompt for the LLM based on the given query and an optional prompt template.
        It retrieves search results using the `rag` object and formats them into the prompt.
        Additional instructions for the LLM are also included in the prompt.

        Args:
            query (str): The query for which to prepare the prompt.
            prompt_template (str, optional): An optional prompt template to use. Defaults to "".

        Returns:
            str: The prepared prompt for the LLM.
        """
        lines = []
        if not prompt_template:
            lines.append("""You are a prompt engineering expert tasked with search results based on relevance.
        Below are the search results with their payloads and scores.
        Search Results:""")
        #self.rag= rag_pipe(url=url, api_key=api_key, collection_name=collection_name)    
        try:
            search_results = self.rag_pipe.invoke(query)
            if search_results and isinstance(search_results, list):
                lines.extend(
                    f"{idx + 1}. ID: {result.id}, Score: {result.score}, Payload: {result.payload['text']}"
                    for idx, result in enumerate(search_results)
                    if result.payload and result.payload['text']
                )
        except Exception as e:
            logging.info(f"An error occurred while preparing the prompt: {e}")
        # Additional instructions
        lines.extend([
            f"query: {query}",
            """Instructions:
            1. Analyze the payloads and score to re-rank the results based on relevance to the query.
            2. Select the top 3 results based on reciprocal rank fusion to summarize the correct answers.
            3. Provide a reason for selecting each of the top 3 results. 
            4. If the result is not relevant, search for other relevant answers within the payloads.
            5. If the result did not contain approximate keywords, search for other relevant answers within the payloads.
            4. The JSON response should include the following structure:
            {
            "top_results" to the query/question: [
                {"id": "result_id_1", "answer": "answer","reason": "reason for selection"},
                {"id": "result_id_2", "answer": "answer","reason": "reason for selection"},
                {"id": "result_id_3", "answer": "answer","reason": "reason for selection"}]
            }
            5. Ensure the JSON is well-structured and valid.
            6. Do not include any other text or explanation outside the JSON format.
            7. The JSON should be formatted with proper indentation for readability.
            8. The answer should be in the context of the question or query asked.
            9. Each answer should be contains not less than 256 tokens.
    """
        ])
        return "\n".join(lines)

    def grounded_generation_from_llm(
        self,
        query: str,
        llm_model_name: str = "mistral-large-latest",
        temperature: float = 0.1,
        max_tokens: Optional[int] = 20000
        ) -> dict:
        """
        Generate response using Mistral model and parse the JSON output.
        
        Args:
            llm_model_name: Name of the LLM model to use (default: "mistral-large-latest")
            Note: The model name should be a valid model available in the Mistral API.
            query: The query to be answered
            
        Returns:
            dict: Parsed JSON response from the LLM
        """
        if not self.llm_client:
            raise ValueError("Mistral client not initialized. Please provide a valid API key.")

        # Generate the prompt for the LLM
        prompt = self.prepare_llm_prompt(query)

        if not prompt:
            raise ValueError("Failed to prepare prompt for LLM.")
        if not llm_model_name:
            raise ValueError("LLM model name is required.")
        if not isinstance(llm_model_name, str):
            raise ValueError("LLM model name should be a string.")
        if not isinstance(prompt, str):
            raise ValueError("Prompt should be a string.")

        chat_response = self.llm_client.chat.complete(
            model=self.llm_model_name or "mistral-large-latest",
            messages=[
                {"role": "system", "content": "You're a helpful assistant. Answer the questions in polite and professional manner."},
                {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0
        )

        # Initialize response with a default value
        json_response = ""
        response_dict = {}

        # Check if chat_response and its attributes are valid
        if (chat_response and chat_response.choices and chat_response.choices[0].message and isinstance(chat_response.choices[0].message.content, str)):
            json_response = chat_response.choices[0].message.content

        # Parse the LLM response
        if json_match := re.search(r'\{.*\}', json_response, re.DOTALL):
            json_part = json_match[0]
            try:
                response_dict = json.loads(json_part)
                # Print the results
                for item in response_dict.get('top_results', []):
                    logger.info(f"ID: {item['id']}, Answer: {item['answer']}, Reason: {item['reason']}")
            except json.JSONDecodeError:
                logger.info("Failed to parse JSON response.")
                return {}
        else:
            print("No valid JSON found in the response.")
            return {}
        return response_dict
    
    
