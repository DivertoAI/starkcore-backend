from runpod.serverless.modules.rp_fastapi import RunpodFastAPI
from main import app

rp_app = RunpodFastAPI(app=app)

@rp_app.run()
def handler(event):
    print("✅ Event received:", event)
    return [
        {
            "id": "1",
            "name": "Debug Bot",
            "age": 23,
            "gender": "Female",
            "image_url": "https://via.placeholder.com/150",
            "personalityDescription": "Testing RunPod output.",
        }
    ]