from runpod.serverless.modules.rp_fastapi import RunpodFastAPI
from main import app

rp_app = RunpodFastAPI(app=app)

@rp_app.run()
def handler(event):
    print("✅ Event received:", event)

    api = event.get("input", {}).get("api", {})
    if api.get("method") == "GET" and api.get("endpoint") == "/characters":
        return [
            {"id": "1", "name": "Test Character", "age": 22, "gender": "Female"}
        ]

    return {"error": "Not found"}