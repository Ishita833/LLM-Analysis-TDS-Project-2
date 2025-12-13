from langchain_core.output_parsers import JsonOutputParser
import time
import requests
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from scraper import get_rendered_html as scraper
from solver_agent import SolverAgent
from llm import OpenRouterLLM
import json
from dotenv import load_dotenv
load_dotenv()

model = "openai/gpt-5.1-codex-max"
api_key = os.getenv("AIPIPE_KEY")

class QuestionTemplate(BaseModel):
   task: str = Field(description="The task that needs to be done")
   files: Dict[str, str] = Field(description="""
                                 Return a key value pair of this format for each of the files that are necessary to accomplish the task
                                 The filename shouldnot contain any slashes '/' or '\\'
                                 <filename>: <download_url>
                                 """) 
   quiz_url: str = Field(description="URL of the quiz")
   submission_url: str= Field(description="""
                              URL to which the answer needs to be submitted, in some cases there won't be a specific submission url mentioned in the quiz page,
                              use the following default submission url: "https://tds-llm-analysis.s-anand.net/submit" 
                              """)
   payload: Dict[str, Any] = Field(description=f"""
                                   Return the payload structure that needs to be submitted to the submission url.
                                   Some quiz url can mention the payload structure but by default this is the following structure.
                                   By default it will be like
                                   {{
                                   "url": <url mentioned in the task>,
                                   "email": {os.getenv("EMAIL")},
                                   "secret": {os.getenv("SECRET")},
                                   "answer": <answer>
                                   }}""")
   other: Optional[str] = Field(description="""
                              This is an optional field which contains information
                              that you think will be useful for the solving agent """)

question_template_parser = JsonOutputParser(pydantic_object=QuestionTemplate)

class SolutionModel(BaseModel):
   quiz_url: str = Field(description="URL of the quiz")
   answer: Any = Field(description="Answer to the task that you inferred")
   reasoning: str = Field(description="Your reasoning to your solution")               # short human-readable summary
   return_payload: Dict[str, Any] = Field(description="The return payload which includes the answer and other metadata that needs to be submitted")

TASK_EXTRACTOR_PROMPT = """
You are a Task Extractor. INPUTS:
- {content}: quiz page text
- {url}: source URL
The task will have two parts first one is finding the answer and the other is to submit the answer using a POST request. The POST request part is only for submission, and is not part of finding the answer.
STRICT INSTRUCTIONS:
1) Produce EXACTLY ONE valid JSON object and NOTHING else (no markdown, no commentary, no code fences).
2) The JSON must contain only these keys (types shown):
{{
  "task": "string",                  // Clear description of the task that needs to be accomplished and submitted to the submission url, don't mention to download the files that are mentioned below, because they are already downloaded.
  "files": {{"<filename>": "url"}},   // full filenames: no '/' or '\\'; return {{}} if none, make sure not to make them malformed. REMEMBER all files will be downloaded in the local system.
  "quiz_url": "string",
  "submission_url": "string",       // URL to POST results; if not present in page, use default below
  "payload": {{"url":"string","email":"string","secret":"string","answer": <answer-format>}}, // STRICTLY follow the answer format mentioned in the question. Submission body shape; fill with defaults if absent
  "other": "string or null"         // optional helper notes but DO NOT include submission_url or payload details here
}}
3) Defaults to use when not present:
   - submission_url: "https://tds-llm-analysis.s-anand.net/submit"
   - payload.email: ENV EMAIL
   - payload.secret: ENV SECRET
4) If a key is absent in the page, return:
   - strings → "" (empty string)
   - files → {{}}
   - other → null
5) "task" field should contain clear instructions on how to solve the quiz and submit the answer.
6) Put any notes about how to solve the task (formatting, edge cases, hints) only in "other" — again, do NOT include submission details there.
7) Filenames must be safe (no slashes) and file values must be URLs when applicable.
8) Do NOT add extra fields or verbose explanations.
9) Retrieve the exact submission url from the url, don't confuse that url with the actual quiz url

QUIZ_CONTENT:
{content}

SOURCE_URL:
{url}
"""

SOLVER_SYSTEM_PROMPT = """
You are a smart quiz solver agent. Your goal is to autonomously solve computational and logic tasks using the provided tools.

**Operating Guidelines:**
1. **Reason Step-by-Step:** Clearly explain your reasoning before executing any tool.
2. **Autonomy:** You have access to tools for downloading files, running Python code, and processing data. Use them proactively rather than asking the user for help.
3. **Resource Management:** If a file mentioned in the task is missing, your first step should be to download it using the `download_file` tool.
4. **Code Execution:** For calculations (like F1 scores, data processing, etc.), always write and execute Python code using `run_code`. Do not attempt to simulate complex math in your head.
5. **Submission:** You must submit the final answer using the `submit_answer` tool to the specific submission URL provided in the task.
6. **AVOID:** Downloading large files and keep the size less than 100 megabytes
"""
SOLVER_USER_PROMPT = """
Here is the specific task you need to complete:
- **Task Description:** {task} 
- **Additional Context:** {other}
- **Required Files:** {files} // All files are already downloaded in the environment, if they are not present download them by yourself.
  *(Note: Check if these files exist. If not, download them immediately.)*
**Submission Details:**
- **Submission URL:** {submission_url}
- **Required Payload Schema:**
```json
  {payload}
```
My Credentials:
Email: {email}
Secret: {secret}
Always include a "tool_call" when returning output
Reference Material:
Original Problem URL: {url} (Use this URL only if you need to verify instructions or find missing data locations.)
Please begin by analyzing the task and executing the necessary steps to solve it. 
**REMEMBER** to always include a "tool_call".
"""

def main(url: str):
    start_time = time.time()
    content = scraper(url)
    llm_task_extractor = OpenRouterLLM(api_key=api_key, model=model, json_mode=True)
    task_extractor_formatted = TASK_EXTRACTOR_PROMPT.format(
        content=content,
        url=url
    )
    task_messages = [
        {"role": "system", "content": task_extractor_formatted}
    ]

    task_metadata = json.loads(llm_task_extractor.invoke(task_messages)["content"])
    print("Task: ", json.dumps(task_metadata, indent=4))
    task = task_metadata.get("task", "")
    other = task_metadata.get("other", "")
    submission_url = task_metadata.get("submission_url", "")
    files_download_url = task_metadata.get("files", "")
    payload = task_metadata.get("payload", "")
    # Download files
    files = []
    for file, download_url in files_download_url.items():
        files.append(download_file(filename=file, url=download_url))

    # Solver agent
    solver_user_prompt = SOLVER_USER_PROMPT.format(task= task,
        other= other,
        files= files,
        url= url,
        email=os.getenv("EMAIL"),
        secret=os.getenv("SECRET"),
        submission_url=submission_url,
        payload=payload)
    conv_history = [
        {
        "role": "system", 
        "content": SOLVER_SYSTEM_PROMPT
    },
        {
        "role": "user", 
        "content": solver_user_prompt
    },
    ]   # list of messages: HumanMessage, AIMessage, ToolMessage
    llm_solver_agent = OpenRouterLLM(api_key=api_key, model=model)
    solver_agent = SolverAgent(llm_solver_agent, conv_history, start_time)
    conv_history = solver_agent.run_agent()
    print(json.dumps(conv_history[-1], indent=4), '\n')
    return solver_agent.next_url

def download_file(url: str, filename: str) -> str:
    """
    Download a file from a URL and save it with the given filename
    in the current working directory.

    Args:
        url (str): Direct URL to the file.
        filename (str): The filename to save the downloaded content as.

    Returns:
        str: Full path to the saved file.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        directory_name = "LLMFiles"
        os.makedirs(directory_name, exist_ok=True)
        parent_directory = os.path.dirname(filename)
        if parent_directory and not os.path.isdir(parent_directory):
            os.makedirs(parent_directory, exist_ok=True)
        path = os.path.join(directory_name, filename)
        print("Path: ", path)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return filename
    except Exception:
        return url

def run_pipeline(url: str):
    while url is not None:
        print("Solving: ", url)
        url = main(url)
