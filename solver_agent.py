from tools import (
    get_rendered_html, download_file,
    run_code, add_dependencies, submit_answer
)
from llm import OpenRouterLLM
import json
import time
from dotenv import load_dotenv
load_dotenv()

RETRY_LIMIT = 2
class SolverAgent:
    def __init__(self, llm: OpenRouterLLM, messages, start_time):
        self.llm = llm
        self.messages = messages 
        self.run_limit = 20
        self.next_url = None
        self.start_time = start_time
        self.retry_count = 0
        self.run = True

    def run_agent(self):
        for _ in range(self.run_limit):
            if not self.run:
                break
            response = self.llm.invoke(self.messages, tools=TOOLS_SCHEMA)
            self.messages.append(response)
            if "tool_calls" in response:
                for tool_call in response["tool_calls"]:
                    self.call_tool(tool_call["function"]["name"], json.loads(tool_call["function"]["arguments"]), tool_call["id"])
            else:
                if not self.run:
                    break
            time.sleep(5)
        return self.messages
    def call_tool(self, func_name, args, id):
        print("Calling tool: ", func_name, "with args", args)
        try:
            if func_name == "submit_answer":
                submission = self.handle_submission(args=args) 
                submission["tool_call_id"] = id
                self.messages.append(submission)
            else:
                result = FUNCTION_MAP[func_name](**args)
                self.messages.append(
                    {
                        "role": "tool",
                        "content": f"Tool call resulted in: {result}",
                        "tool_call_id": id
                    }
                )
        except Exception as e:
            result = {"error": f"Error {e} occurred while calling {func_name} with args {args}"}
            self.messages.append(
                {
                    "role": "tool",
                    "content": f"Tool call resulted in: {result}",
                    "tool_call_id": id
                }
            )

    def handle_submission(self, args):
        try:
            self.retry_count += 1
            resp = submit_answer(**args)
            correct = resp.get("correct", None)
            self.next_url = resp.get("next_url", None)
            reason = resp.get("reason", None)
            if correct or (self.retry_count > RETRY_LIMIT and self.next_url is not None):
                result = {
                    "role": "tool",
                    "content": "Task completed successfully you can stop!",
				        }
                self.run = False
            elif not self.next_url:
                result = {
                        "role": "tool",
                        "content": f"Retry again! Your previous answer was wrong because, {reason}",
                    }
            elif time.time() - self.start_time >= 180:
                result = {
                       "role": "tool",
                       "content": "Task completed",
					}
                self.run = False
            else:
                result = {
                    "role": "tool",
                    "content": f"Retry again! Your previous answer was wrong because, {reason}",
                }
                    
        except Exception as e:
            result = {
                "role":"tool",
                "content": f"Error: occurred while using the tool 'submit_answer' using {args, {e}}",
            }
        return result
			

FUNCTION_MAP = {
    "get_rendered_html": get_rendered_html,
    "run_code": run_code,
    "download_file": download_file,
    "add_dependencies": add_dependencies,
	"submit_answer": submit_answer
}

TOOLS_SCHEMA = [
  {
    "type": "function",
    "function": {
      "name": "get_rendered_html",
      "description": "Fetch a webpage, render it with a headless browser, and return its HTML and image URLs.",
      "parameters": {
        "type": "object",
        "properties": {
          "url": { "type": "string", "description": "The URL to fetch and render." }
        },
        "required": ["url"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "run_code",
      "description": "Execute Python code inside an isolated environment and return stdout/stderr. **DO NOT** use this tool to submit the answer",
      "parameters": {
        "type": "object",
        "properties": {
          "code": { "type": "string", "description": "Python source code to execute." }
        },
        "required": ["code"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "download_file",
      "description": "Download a file from a URL and save it with the given filename.",
      "parameters": {
        "type": "object",
        "properties": {
          "url": { "type": "string", "description": "Direct URL to the file." },
          "filename": { "type": "string", "description": "Filename to save the content as." }
        },
        "required": ["url", "filename"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "add_dependencies",
      "description": "Install Python dependencies using 'uv add'.",
      "parameters": {
        "type": "object",
        "properties": {
          "dependencies": {
            "type": "array",
            "items": { "type": "string" },
            "description": "List of pip package names to install."
          }
        },
        "required": ["dependencies"]
      }
    }
  },
{
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the answer using this tool. After you found the answer use this tool to submit and validate.",
        "parameters": {
            "type": "object",
            "properties": {
                "submission_url": {
                    "type": "string",
                    "description": "The URL where the final answer should be submitted."
                },
                "payload": {
                    "type": "string", 
                    "description": "The submission data as a valid JSON string. Example: '{\"url\": \"...\", \"answer\": ...}'"
                }
            },
            "required": ["submission_url", "payload"]
        }
    }
}
]
