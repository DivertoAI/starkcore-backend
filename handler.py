import os
import uuid
import torch
from PIL import Image
from io import BytesIO
import requests
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.models import UNet2DConditionModel
from peft import PeftModel, PeftConfig

# Load .env
load_dotenv(dotenv_path=".env.local")
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
MODEL_ID = "RunDiffusion/Juggernaut-XL-v8"
device = "cuda"

# Login to HF
if HF_TOKEN:
    login(HF_TOKEN)

# Load pipelines once (global)
txt2img_pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None
).to(device)
txt2img_pipe.enable_xformers_memory_efficient_attention()

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None
).to(device)
img2img_pipe.enable_xformers_memory_efficient_attention()

def apply_lora_if_any(lora_repo: str):
    if not lora_repo:
        return

    unet_base = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet",
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.float16
    )
    lora_config = PeftConfig.from_pretrained(lora_repo, use_auth_token=HF_TOKEN)
    unet_lora = PeftModel.from_pretrained(unet_base, lora_repo, use_auth_token=HF_TOKEN)
    unet_lora = unet_lora.merge_and_unload()

    txt2img_pipe.unet = unet_lora
    img2img_pipe.unet = unet_lora

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def save_image_temp(image: Image.Image) -> str:
    path = f"/tmp/{uuid.uuid4().hex}.png"
    image.save(path)
    return path

def handler(event):
    try:
        mode = event.get("mode", "txt2img")
        prompt = event.get("prompt", "masterpiece, high detail")
        guidance = float(event.get("guidance_scale", 7.5))
        steps = max(5, int(event.get("steps", 30)))
        strength = float(event.get("strength", 0.75))
        lora = event.get("lora_repo", None)

        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        if lora:
            apply_lora_if_any(lora)

        default_suffix = ", ultra realistic, soft shadows, cinematic lighting"
        image_paths = []

        for p in prompts:
            enriched_prompt = f"{p}{default_suffix}" if "realistic" not in p.lower() else p

            if mode == "img2img":
                url = event.get("image_url")
                if not url:
                    return {"error": "Missing 'image_url' for img2img"}
                init_img = load_image_from_url(url)
                image = img2img_pipe(
                    prompt=enriched_prompt,
                    image=init_img,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=steps
                ).images[0]
            else:
                image = txt2img_pipe(
                    prompt=enriched_prompt,
                    guidance_scale=guidance,
                    num_inference_steps=steps
                ).images[0]

            image_paths.append(save_image_temp(image))

        return {"image_paths": image_paths}

    except Exception as e:
        return {"error": str(e)}