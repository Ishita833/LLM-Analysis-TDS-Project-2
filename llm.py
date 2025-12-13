import httpx
import time
import json
import pprint
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from typing import Dict, Any, List

class OpenRouterLLM:
    def __init__(self, api_key: str, model: str, json_mode=False):
        self.api_key = api_key
        self.model = model
        self.url = "https://aipipe.org/openrouter/v1/chat/completions"
        # self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.json_mode = json_mode 

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        with httpx.Client(timeout=60) as client:
            try:
                res = client.post(self.url, headers=headers, json=payload)
                res.raise_for_status()
                return res.json()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 413:
                    return {"error": 413, "reason": e.response.text}
                    # Handle it however you want:
                else:
                    return {"error": status, "reason": e.response.text}
 
    def invoke(self, messages, tools=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "tool_call": "required"
        }
        if tools:
            payload["tools"] = tools
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = self._post(payload)
        if "choices" not in response:
            print(response)
        while "error" in response:
            print("Retrying...", response["error"], response["reason"])
            if response["error"] == 429:
                time.sleep(20)
                continue
            payload["messages"].append(
                {
                    "role": "user",
                    "content": "Your last response is malformed"
                }
            )
            response = self._post(payload)

        return response["choices"][0]["message"]