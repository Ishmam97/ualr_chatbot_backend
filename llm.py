import requests
import json

def call_gemini(api_key: str, prompt: str, model: str = "gemini-1.5-flash-latest", system_prompt: str = None):
    """
    Calls the Google Gemini API to generate content based on a prompt.

    Args:
        api_key: Your Google AI API key.
        prompt: The user prompt for the model.
        model: The Gemini model to use (e.g., "gemini-1.5-flash-latest", "gemini-1.0-pro").
               See https://ai.google.dev/models/gemini for available models.
        system_prompt: Optional system instruction for the model. Note: Support and exact
                       implementation can vary slightly between models/API versions.
                       The v1beta API often uses 'system_instruction'.

    Returns:
        The generated text response from the model as a string.

    Raises:
        requests.exceptions.RequestException: If the API request fails (network issue, bad status code).
        KeyError: If the response JSON format is unexpected.
        IndexError: If the response JSON format is unexpected (e.g., missing 'candidates').
        ValueError: If the API returns an error structure or blocks the request.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    # Basic payload structure for Gemini
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
        # "generationConfig": {
        #     "temperature": 0.7,
        #     "maxOutputTokens": 1024,
        # }
    }

    if system_prompt:
        payload["system_instruction"] = {
             "parts": [{"text": system_prompt}]
        }

    headers = {'Content-Type': 'application/json'}
    params = {'key': api_key} # API key passed as a query parameter

    try:
        response = requests.post(api_url, headers=headers, json=payload, params=params)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()

        # --- Robust Response Parsing ---
        # Check for top-level API errors first
        if 'error' in response_data:
            raise ValueError(f"Gemini API Error: {response_data['error'].get('message', 'Unknown error')}")

        # Check for candidates and potential blocking reasons
        if 'candidates' not in response_data or not response_data['candidates']:
            # Check if the prompt was blocked
            if 'promptFeedback' in response_data and 'blockReason' in response_data['promptFeedback']:
                 block_reason = response_data['promptFeedback']['blockReason']
                 safety_ratings = response_data['promptFeedback'].get('safetyRatings', [])
                 raise ValueError(f"Request blocked due to: {block_reason}. Safety Ratings: {safety_ratings}")
            else:
                 raise KeyError("Response missing 'candidates' and no clear block reason found.")

        # Extract text from the first candidate
        candidate = response_data['candidates'][0]

        # Check if the candidate content exists
        if 'content' not in candidate or 'parts' not in candidate['content'] or not candidate['content']['parts']:
             # Check for finish reason (like safety)
             finish_reason = candidate.get('finishReason', 'UNKNOWN')
             safety_ratings = candidate.get('safetyRatings', [])
             raise ValueError(f"Candidate missing content parts. Finish Reason: {finish_reason}. Safety Ratings: {safety_ratings}")

        # Extract the text
        generated_text = candidate['content']['parts'][0].get('text')
        if generated_text is None:
            raise KeyError("First part in candidate content is missing 'text'.")

        return generated_text

    except requests.exceptions.RequestException as e:
        # Log or handle HTTP errors
        print(f"API Request failed: {e}")
        # Try to get more details from the response body if possible
        try:
            error_details = response.json()
            print(f"Error Response Body: {json.dumps(error_details, indent=2)}")
        except (AttributeError, ValueError, json.JSONDecodeError): # Handle cases where response doesn't exist or isn't JSON
            pass
        raise  # Re-raise the exception
    except (KeyError, IndexError, ValueError) as e:
        # Log or handle parsing/value errors
        print(f"Error processing Gemini response: {e}")
        try:
            # Log the problematic response data for debugging
            print(f"Problematic Response Data: {json.dumps(response_data, indent=2)}")
        except NameError: # response_data might not be defined if request failed early
             pass
        raise # Re-raise the specific error


def call_ollama(prompt, model="qwen2.5:7b", system_prompt=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    response.raise_for_status()
    return response.json()["response"]