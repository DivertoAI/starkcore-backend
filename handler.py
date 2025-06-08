import os
import torch
import requests
import uuid
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.models import UNet2DConditionModel
from peft import PeftModel, PeftConfig

# Load .env.local for HuggingFace token
load_dotenv(dotenv_path=".env.local")
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
MODEL_ID = "RunDiffusion/Juggernaut-XL-v8"
DEVICE = "cuda"

# Authenticate with Hugging Face
if HF_TOKEN:
    login(HF_TOKEN)

def load_pipeline_with_lora(model_id: str, lora_repo: str = None):
    """Load txt2img and img2img pipelines, optionally with LoRA support."""
    txt2img = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_auth_token=HF_TOKEN,
        use_safetensors=True,
        safety_checker=None
    ).to(DEVICE)
    txt2img.enable_xformers_memory_efficient_attention()

    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_auth_token=HF_TOKEN,
        use_safetensors=True,
        safety_checker=None
    ).to(DEVICE)
    img2img.enable_xformers_memory_efficient_attention()

    if lora_repo:
        print(f"🔗 Loading LoRA: {lora_repo}")
        unet_base = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=torch.float16,
            use_auth_token=HF_TOKEN,
        )
        lora_config = PeftConfig.from_pretrained(lora_repo, use_auth_token=HF_TOKEN)
        unet_lora = PeftModel.from_pretrained(unet_base, lora_repo, use_auth_token=HF_TOKEN)
        unet_lora = unet_lora.merge_and_unload()

        txt2img.unet = unet_lora
        img2img.unet = unet_lora

    return txt2img, img2img

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def save_image_to_tmp(image: Image.Image) -> str:
    path = f"/tmp/{uuid.uuid4().hex}.png"
    image.save(path, format="PNG")
    return path

def handler(event: dict) -> dict:
    """
    RunPod handler entrypoint.

    Input format:
    {
        "mode": "txt2img" | "img2img",
        "prompt": "...",
        "image_url": "...",        # required for img2img
        "strength": 0.6,           # optional
        "guidance_scale": 7.5,
        "steps": 30,
        "lora_repo": "user/lora"   # optional
    }

    Output format:
    {
        "image_paths": ["/tmp/abc.png"]
    }
    """
    try:
        mode = event.get("mode", "txt2img")
        prompts = event.get("prompt", "masterpiece, high detail")
        if isinstance(prompts, str):
            prompts = [prompts]

        strength = float(event.get("strength", 0.75))
        guidance_scale = float(event.get("guidance_scale", 7.5))
        steps = max(5, int(event.get("steps", 30)))
        lora_repo = event.get("lora_repo")

        txt2img, img2img = load_pipeline_with_lora(MODEL_ID, lora_repo)

        image_paths = []
        for prompt in prompts:
            enriched = f"{prompt}, ultra realistic, cinematic lighting, 4k, soft shadows"
            if mode == "img2img":
                image_url = event.get("image_url")
                if not image_url:
                    return {"error": "Missing 'image_url' for img2img mode"}
                init_image = load_image_from_url(image_url)
                result = img2img(
                    prompt=enriched,
                    image=init_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps
                ).images[0]
            else:
                result = txt2img(
                    prompt=enriched,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps
                ).images[0]

            path = save_image_to_tmp(result)
            image_paths.append(path)

        return {"image_paths": image_paths}
    
    except Exception as e:
        return {"error": str(e)}