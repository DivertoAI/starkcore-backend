from runpod.serverless.modules.rp_handler import RunpodHandler
from handler import handler

# Start the serverless handler loop
RunpodHandler(handler).start()