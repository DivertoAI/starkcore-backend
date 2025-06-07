import subprocess
import time
from runpod.serverless import start
import requests

# Start FastAPI in the background
subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
time.sleep(2)  # wait for FastAPI to start

def handler(event):
    print("✅ Event received:", event)

    api = event.get("input", {}).get("api")
    if not api:
        return {"error": "Missing 'api' in input"}

    endpoint = api.get("endpoint", "")
    method = api.get("method", "GET").upper()
    payload = api.get("body", None)

    try:
        url = f"http://127.0.0.1:8000{endpoint}"

        if method == "GET":
            res = requests.get(url)
        elif method == "POST":
            res = requests.post(url, json=payload)
        else:
            return {"error": f"Unsupported method {method}"}

        return {
            "status_code": res.status_code,
            "data": res.json()
        }

    except Exception as e:
        return {"error": str(e)}

start({"handler": handler})