from runpod.serverless import start
from handler import handler  # Assuming handler.py is in root

start({"handler": handler})