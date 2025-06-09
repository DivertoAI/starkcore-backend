import os
import torch
from diffusers import StableDiffusionPipeline
from runpod.serverless import start
from dotenv import load_dotenv

# Load environment variables from .env.local (if running locally)
load_dotenv(dotenv_path=".env.local", override=True)

# Hugging Face token and model config
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
MODEL_REPO = "RunDiffusion/Juggernaut-XL-v8"

# ✅ RunPod volume mount path
MODEL_PATH = "/runpod-volume/juggernaut-xl"

# Device selection (safe fallback)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure volume directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# 🔁 Download model to /runpod-volume if not already cached
def download_model_if_needed():
    expected_file = os.path.join(MODEL_PATH, "model_index.json")
    if not os.path.exists(expected_file):
        print("🔽 Downloading model to /runpod-volume...")
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_REPO,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16,
            safety_checker=None,
            cache_dir=MODEL_PATH
        )
        pipe.save_pretrained(MODEL_PATH)
        del pipe
    else:
        print("✅ Model already cached in /runpod-volume")

download_model_if_needed()

# ✅ Load model from cached path
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.enable_xformers_memory_efficient_attention()

# 🚀 Main handler
def handler(event):
    try:
        prompt = event.get("prompt", "masterpiece, beautiful girl, cinematic lighting")
        guidance = float(event.get("guidance_scale", 7.5))
        steps = int(event.get("steps", 30))

        image = pipe(prompt, guidance_scale=guidance, num_inference_steps=steps).images[0]
        out_path = "/tmp/output.png"
        image.save(out_path)

        return {"image_paths": [out_path]}
    except Exception as e:
        return {"error": str(e)}

# 🧠 Start RunPod serverless handler
start({"handler": handler})