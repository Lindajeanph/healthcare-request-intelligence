"""Run only after local app is listening on 127.0.0.1:8000."""
import json
import requests

url = "http://127.0.0.1:8000/triage"
payload = {"text": "I need to move my appointment from Monday to Thursday afternoon."}
response = requests.post(url, json=payload, timeout=120)
response.raise_for_status()
print(json.dumps(response.json(), indent=2))
