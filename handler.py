import os
import torch
import traceback
from diffusers import StableDiffusionPipeline
from runpod.serverless import start
from dotenv import load_dotenv
import warnings

# Load env vars (locally)
load_dotenv(dotenv_path=".env.local", override=True)

# Hugging Face auth
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
MODEL_REPO = "RunDiffusion/Juggernaut-XL-v8"
MODEL_PATH = "/runpod-volume/juggernaut-xl"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Optional: suppress safety checker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

# Ensure mount path exists
os.makedirs(MODEL_PATH, exist_ok=True)

# Download model if needed
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
        print("✅ Model already cached at /runpod-volume")

download_model_if_needed()

# Load pipeline from cached model
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

# Try enabling xformers if supported
try:
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        print("⚡ xformers memory-efficient attention enabled.")
except Exception as e:
    print(f"⚠️ Could not enable xformers: {e}")

# RunPod handler
def handler(event):
    try:
        input_data = event.get("input", {})  # 💡 always use 'input' key with fallback
        prompt = input_data.get("prompt", "masterpiece, beautiful girl, cinematic lighting")
        guidance = float(input_data.get("guidance_scale", 7.5))
        steps = int(input_data.get("steps", 30))

        print(f"🎨 Generating: '{prompt}' | steps: {steps}, scale: {guidance}")
        image = pipe(prompt, guidance_scale=guidance, num_inference_steps=steps).images[0]

        out_path = "/tmp/output.png"
        image.save(out_path)

        print("✅ Generation complete.")
        return {"image_paths": [out_path]}

    except Exception as e:
        print("❌ Error during generation:", e)
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }

# 🔁 Start serverless worker
start({"handler": handler})