import requests
import os

API_URL = "http://localhost:8000/query"
API_KEY = os.getenv("GOOGLE_API_KEY") or input("Enter your Google API Key: ")

test_cases = [
    {
        "query": "Who is the graduate coordinator for Accounting?",
        "expected": "Sonya Premeaux"
    },
    {
        "query": "Who is Mark funk tell me about him",
        "expected": "AEAX"
    },
     {
        "query": "Can I get the contact info for the coordinator of Adult Education?",
        "expected": "Jennifer Holtz"
    }
]

def test_query(query, expected_substring):
    print(f"\n Query: {query}")
    response = requests.post(API_URL, json={"query": query, "api_key": API_KEY})
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    data = response.json()
    result = data.get("response", {}).get("content", "")
    print(f"Response: {result}")

    if expected_substring.lower() in result.lower():
        print("✅ Match")
    else:
        print(f" Expected to find: '{expected_substring}'")

if __name__ == "__main__":
    for case in test_cases:
        test_query(case["query"], case["expected"])
