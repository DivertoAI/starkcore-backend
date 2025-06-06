from runpod.serverless import start
import requests

API_BASE = "http://127.0.0.1:8000"  # FastAPI runs inside the same container

def handler(event):
    print("✅ Event received:", event)

    api = event.get("input", {}).get("api")
    if not api:
        return {"error": "Missing 'api' in input"}

    endpoint = api.get("endpoint", "")
    method = api.get("method", "GET").upper()
    payload = api.get("body", None)

    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=payload)
        else:
            return {"error": f"Unsupported method {method}"}

        return {
            "status_code": response.status_code,
            "data": response.json()
        }

    except Exception as e:
        return {"error": str(e)}

start({"handler": handler})