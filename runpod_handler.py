from runpod.serverless import start
from handler import handler as ai_handler  # Import your AI handler

def entrypoint(event):
    print("✅ Event received:", event)
    
    try:
        response = ai_handler(event)
        return { "output": response }
    except Exception as e:
        print("❌ Error:", str(e))
        return { "error": str(e) }

start({"handler": entrypoint})