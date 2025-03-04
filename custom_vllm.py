from langchain.llms.base import LLM
import requests
from typing import Optional, List, Mapping, Any

class CustomVLLM(LLM):
    """Custom LLM client for vLLM server mode without external dependencies."""
    server_endpoint: str = "http://localhost:8000/v1"
    model_name: str = "TheBloke/Llama-2-13B-chat-GPTQ"
    timeout: int = 30

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 512,  # High enough to not interfere, or omit
            "temperature": 0.7,
            "stop": stop if stop else ["\n"]  # Default to sentence end
        }
        try:
            response = requests.post(
                f"{self.server_endpoint}/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as e:
            raise ValueError(f"Failed to reach vLLM server: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "custom_vllm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"server_endpoint": self.server_endpoint, "model_name": self.model_name}