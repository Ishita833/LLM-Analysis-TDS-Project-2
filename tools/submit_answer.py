import httpx
import json

def submit_answer(submission_url: str, payload: dict | str):
    """
    Submits a JSON payload to a given URL via POST.
    
    Args:
        submission_url (str): The destination URL for the submission.
        payload (dict): The data to send as JSON.
        headers (dict, optional): Additional HTTP headers.

    Returns:
        dict: A structured result containing response details or error info.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {"correct": False, "reason": "Failed to parse payload string as JSON.", "next_url": None}
    # Ensure headers always exist
    final_headers = {"Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=20) as client:
            response = client.post(
                submission_url,
                json=payload,
                headers=final_headers
            )

        # 1. Check for HTTP errors FIRST
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # If the server sent a helpful error message in the body, try to get it
            return {
                "correct": False, 
                "next_url": None,
                "reason": f"HTTP Error {e.response.status_code}: {e.response.text}" 
            }

        # 2. SAFELY attempt to parse JSON
        try:
            result = response.json()
            print("Got the response: ", result) 

        except json.JSONDecodeError:
            # Handle cases where success (200 OK) returns non-JSON (rare but possible)
            return {
                "correct": False,
                "next_url": None,
                "reason": f"Server returned invalid JSON: {response.text}"
            }

        # 3. Process the valid JSON result
        correct = result.get("correct")
        next_url = result.get("url")
        reason = result.get("reason", "")
        return {
            "correct": correct, 
            "next_url": next_url,
            "reason": reason
        }

    except Exception as e:
        return {
            "correct": False, 
            "next_url": None,
            "reason": str(e)
        }

