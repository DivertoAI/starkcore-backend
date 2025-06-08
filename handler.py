import torch
import os
from diffusers import StableDiffusionPipeline
from runpod.serverless import start

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
device = "cuda"

# 🔁 Point to your network volume
MODEL_PATH = "/vol/juggernaut-xl"

def download_and_cache_model():
    """Download model to network volume once if not exists."""
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model to /vol...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "RunDiffusion/Juggernaut-XL-v8",
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16,
            safety_checker=None,
            cache_dir=MODEL_PATH  # Important!
        )
        pipe.save_pretrained(MODEL_PATH)
        del pipe
    else:
        print("✅ Model already cached in /vol")

download_and_cache_model()

# ✅ Load model from /vol
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)
pipe.enable_xformers_memory_efficient_attention()

def handler(event):
    try:
        prompt = event.get("prompt", "masterpiece, beautiful girl, cinematic lighting")
        guidance = float(event.get("guidance_scale", 7.5))
        steps = int(event.get("steps", 30))

        image = pipe(prompt, guidance_scale=guidance, num_inference_steps=steps).images[0]
        out_path = f"/tmp/output.png"
        image.save(out_path)
        return {"image_paths": [out_path]}
    except Exception as e:
        return {"error": str(e)}

start({"handler": handler})