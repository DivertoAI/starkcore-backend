# handler.py
from runpod.serverless.modules.rp_fastapi import RunpodFastAPI
from main import app

# Optional: define extra logic before/after the request
# This is where you'd wrap logic if needed
rp_app = RunpodFastAPI(app=app)

@rp_app.run()
def handler(event):
    # Optionally inspect event["input"] to call specific logic
    return {"message": "FastAPI backend is live"}