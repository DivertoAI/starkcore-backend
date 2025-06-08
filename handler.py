# handler.py
import torch
import os
from diffusers import StableDiffusionPipeline
from runpod.serverless import start
from huggingface_hub import login

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
MODEL_ID = "RunDiffusion/Juggernaut-XL-v8"
device = "cuda"

# Login and preload model ONCE
if HF_TOKEN:
    login(HF_TOKEN)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
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