import os
import torch
import traceback
import warnings
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline
from runpod.serverless import start

# Load environment variables (for local dev)
load_dotenv(dotenv_path=".env.local", override=True)

# Constants
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
MODEL_REPO = "RunDiffusion/Juggernaut-XL-v8"
MODEL_PATH = "/runpod-volume/juggernaut-xl"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Suppress optional warnings
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

# Ensure model volume directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# 🔁 Download model if needed
def download_model_if_needed():
    model_index_file = os.path.join(MODEL_PATH, "model_index.json")
    if not os.path.exists(model_index_file):
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
        print("✅ Model already cached at /runpod-volume")

download_model_if_needed()

# ✅ Load model
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

# Enable xformers if available
try:
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        print("⚡ xformers memory-efficient attention enabled.")
except Exception as e:
    print(f"⚠️ Could not enable xformers: {e}")

# 🚀 Main handler
def handler(event):
    try:
        input_data = event.get("input", {})
        if not isinstance(input_data, dict):
            raise ValueError("Expected 'input' to be a dictionary.")

        prompt = input_data.get("prompt", "masterpiece, beautiful girl, cinematic lighting")
        guidance = float(input_data.get("guidance_scale", 7.5))
        steps = int(input_data.get("steps", 30))

        print(f"🎨 Prompt: '{prompt}' | Steps: {steps} | Guidance: {guidance}")

        image = pipe(prompt, guidance_scale=guidance, num_inference_steps=steps).images[0]
        out_path = "/tmp/output.png"
        image.save(out_path)

        print("✅ Image generation successful.")
        return {"image_paths": [out_path]}

    except Exception as e:
        print("❌ Error during generation:")
        traceback.print_exc()
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }

# Start RunPod serverless handler
start({"handler": handler})