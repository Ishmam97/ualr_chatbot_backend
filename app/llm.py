import requests
import json
from langchain_google_genai import ChatGoogleGenerativeAI

def call_gemini(api_key: str, prompt: str, model: str = "gemini-1.5-flash-latest", system_prompt: str = None):
    try:
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.2,
            max_output_tokens=1024,
            api_key=api_key,
        )
        # Format messages with 'role' and 'content' keys
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = llm.invoke(messages)

        # Check for block reason in the response
        if response.response_metadata.get("prompt_feedback", {}).get("block_reason") != 0:
            raise ValueError(f"Request blocked: {response.response_metadata['prompt_feedback']['block_reason']}")
        safety_ratings = response.response_metadata.get("safety_ratings", [])
        if any(rating.get("blocked") for rating in safety_ratings):
            raise ValueError(f"Request blocked by safety ratings: {safety_ratings}")
        if not response.content:
            raise ValueError("Empty response content")

        return response
        
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"API request failed: {str(e)}")
    except KeyError as e:
        raise KeyError(f"Unexpected response format: {str(e)}")
    except IndexError as e:
        raise IndexError(f"Unexpected response format: {str(e)}")
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