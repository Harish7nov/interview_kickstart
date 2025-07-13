import requests

# URL of your FastAPI endpoint
url = "http://localhost:8000/generate"

# Payload with prompt and optional parameters
payload = {
    "query": "Catch-up with Alex, Sam, and Ella on May 20, 2024, 10am (30 minutes)",
    "max_new_tokens": 200
}

# Send POST request
response = requests.post(url, json=payload)

# Print the generated response
if response.status_code == 200:
    print("Generated Text:", response.json()["response"])
else:
    print("Error:", response.text)
