import os
import torch
from diffusers import StableDiffusionPipeline
from runpod.serverless import start

# Path to your model inside the RunPod-mounted volume
MODEL_PATH = "/volumes/models/Juggernaut-XL-v8"
device = "cuda"

# Load the model from local disk (volume) without using Hugging Face token
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
        out_path = "/tmp/output.png"
        image.save(out_path)

        return {"image_paths": [out_path]}

    except Exception as e:
        return {"error": str(e)}

start({"handler": handler})