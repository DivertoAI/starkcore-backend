from runpod.serverless.modules.rp_handler import RunpodHandler
from app.ai.handler import handler

# Start the serverless handler loop
RunpodHandler(handler).start()