"""LLM Provider abstractions for different AI models"""

import requests
import json
import time
from typing import Dict, Any, Optional
import streamlit as st


class NvidiaAPI:
    """NVIDIA API provider for multiple models with fallback support"""
    
    # Models ordered by speed/efficiency (fastest first)
    AVAILABLE_MODELS = [
        "mistralai/mistral-small-4-119b-2603",  # Hybrid MoE, fastest
        "mistralai/mistral-large",               # Fallback
        "meta/llama-2-70b-chat-hf",              # Meta Llama
        "google/gemma-4-31b-it",                 # Original choice (slowest but powerful)
    ]
    
    def __init__(self, api_key: str, model_name: str = None):
        """Initialize NVIDIA API client
        
        Args:
            api_key: NVIDIA API key (Bearer token)
            model_name: Optional specific model to use
        """
        self.api_key = api_key
        self.invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.max_retries = 3
        self.timeout = 60
        
        # Set model
        if model_name and model_name in self.AVAILABLE_MODELS:
            self.model = model_name
        else:
            self.model = self.AVAILABLE_MODELS[0]  # Default to fastest model
        
        self.model_name = self.model.split('/')[-1]
        self.current_model_index = self.AVAILABLE_MODELS.index(self.model)
    
    def generate_response(self, prompt: str, temperature: float = 0.5, max_tokens: int = 1024, stream: bool = False) -> str:
        """Generate response using NVIDIA API with model fallback
        
        Args:
            prompt: The prompt/question to send to the model
            temperature: Temperature for response generation (0.0-2.0)
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response
            
        Returns:
            Generated text response
        """
        # Try current model first, then fallback to others
        attempted_models = []
        
        # Start with current model
        models_to_try = [self.model] + [m for m in self.AVAILABLE_MODELS if m != self.model]
        
        for model_attempt in models_to_try:
            attempted_models.append(model_attempt)
            response = self._call_api_with_retries(model_attempt, prompt, temperature, max_tokens, stream, attempted_models)
            
            # If successful, update current model
            if response and not response.startswith("⚠️") and not response.startswith("🔄") and not response.startswith("⏳") and not response.startswith("🔐"):
                self.model = model_attempt
                self.model_name = model_attempt.split('/')[-1]
                return response
        
        # All models failed
        return "🚨 All NVIDIA models are currently unavailable. Please try again in a few moments."
    
    def _call_api_with_retries(self, model: str, prompt: str, temperature: float, max_tokens: int, stream: bool, attempted_models: list) -> str:
        """Call NVIDIA API with specified model and retry logic"""
        for attempt in range(self.max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "text/event-stream" if stream else "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stream": stream,
                }
                
                response = requests.post(
                    self.invoke_url,
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                if stream:
                    return self._handle_streaming_response(response)
                else:
                    return self._handle_json_response(response)
            
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    if len(attempted_models) == 1:  # Only show if first attempt
                        st.warning(f"⏱️ {model.split('/')[-1]} timeout (attempt {attempt + 1}/{self.max_retries}). Retrying...")
                    time.sleep(wait_time)
                else:
                    # Try next model
                    if model != self.AVAILABLE_MODELS[-1]:
                        st.info(f"⏭️ Switching from {model.split('/')[-1]} to faster model...")
                    return None  # Signal to try next model
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 504:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        if len(attempted_models) == 1:
                            st.warning(f"🔄 Gateway timeout (attempt {attempt + 1}/{self.max_retries}). Retrying...")
                        time.sleep(wait_time)
                    else:
                        if model != self.AVAILABLE_MODELS[-1]:
                            st.info(f"⏭️ Switching from {model.split('/')[-1]} to faster model...")
                        return None
                elif e.response.status_code == 429:
                    # Rate limited, try next model
                    if model != self.AVAILABLE_MODELS[-1]:
                        st.info(f"⏭️ {model.split('/')[-1]} rate limited. Trying faster model...")
                    return None
                elif e.response.status_code == 401:
                    return "🔐 Authentication failed. Please check your NVIDIA API key."
                else:
                    if model != self.AVAILABLE_MODELS[-1]:
                        st.info(f"⏭️ Model error. Trying next model...")
                    return None
            
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    if model != self.AVAILABLE_MODELS[-1]:
                        st.info(f"⏭️ Switching to faster model...")
                    return None
            
            except Exception as e:
                if model != self.AVAILABLE_MODELS[-1]:
                    return None
                return f"Unexpected error: {str(e)}"
        
        return None
    
    def _handle_json_response(self, response: requests.Response) -> str:
        """Handle non-streaming JSON response"""
        try:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            return "No response received from model"
        except json.JSONDecodeError:
            return f"Failed to parse response: {response.text}"
    
    def _handle_streaming_response(self, response: requests.Response) -> str:
        """Handle streaming response"""
        full_response = ""
        try:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    
                    # Skip SSE event metadata
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        
                        # Skip [DONE] marker
                        if data_str == "[DONE]":
                            continue
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    full_response += content
                        except json.JSONDecodeError:
                            continue
            
            return full_response if full_response else "No content received from streaming response"
        except Exception as e:
            return f"Error processing stream: {str(e)}"
    
    def test_connection(self) -> bool:
        """Test API connection with available models"""
        for model in self.AVAILABLE_MODELS:
            for attempt in range(2):
                try:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/json"
                    }
                    
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 50,
                        "temperature": 0.5,
                        "stream": False,
                    }
                    
                    response = requests.post(
                        self.invoke_url,
                        headers=headers,
                        json=payload,
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        self.model = model
                        self.model_name = model.split('/')[-1]
                        st.success(f"✅ Connected to {self.model_name}")
                        return True
                except requests.exceptions.Timeout:
                    if attempt == 0:
                        time.sleep(2)
                        continue
                except Exception:
                    continue
        
        st.error("❌ Could not connect to any NVIDIA API model")
        return False
