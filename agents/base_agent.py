import logging
import sqlite3
import requests
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, config, db_path, ollama_base_url):
        self.config = config
        self.db_conn = sqlite3.connect(db_path)
        self.ollama_url = ollama_base_url
        self.llm_model = config.get("llm_model", "llama3")
        self.message_bus = None
        
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def connect_to_message_bus(self, message_bus):
        """Connect to the inter-agent communication bus"""
        self.message_bus = message_bus
    
    def _call_ollama_api(self, prompt):
        """Call Ollama API to get LLM response"""
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.get("temperature", 0.2),
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                self.logger.error(f"API call failed with status code {response.status_code}: {response.text}")
                return ""
        except Exception as e:
            self.logger.error(f"Error calling Ollama API: {e}")
            return ""
    
    @abstractmethod
    def run_periodic_tasks(self):
        """Run periodic tasks for this agent"""
        pass