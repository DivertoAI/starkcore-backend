from runpod.serverless.modules.rp_fastapi import RunpodFastAPI
from main import app
from handler import handler as ai_handler

rp_app = RunpodFastAPI(app=app)

@rp_app.run()
def handler(event):
    return ai_handler(event)