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

# Load environment variables
load_dotenv(dotenv_path=".env.local")
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
MODEL_ID = "RunDiffusion/Juggernaut-XL-v8"
device = "cuda"

if HF_TOKEN:
    login(HF_TOKEN)

def load_pipeline_with_lora(model_id: str, lora_repo: str = None):
    """Load SD pipelines and optionally apply a LoRA."""
    txt2img = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None
    ).to(device)
    txt2img.enable_xformers_memory_efficient_attention()

    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None
    ).to(device)
    img2img.enable_xformers_memory_efficient_attention()

    if lora_repo:
        print(f"Loading LoRA: {lora_repo}")
        unet_base = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16
        )
        lora_config = PeftConfig.from_pretrained(lora_repo, use_auth_token=HF_TOKEN)
        unet_lora = PeftModel.from_pretrained(unet_base, lora_repo, use_auth_token=HF_TOKEN)
        unet_lora = unet_lora.merge_and_unload()
        txt2img.unet = unet_lora
        img2img.unet = unet_lora

    return txt2img, img2img

def load_image_from_url(url: str) -> Image.Image:
    """Load an image from a URL for img2img use."""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def save_image_to_tmp_and_get_path(image: Image.Image) -> str:
    """Save image to temporary file and return its path."""
    file_name = f"/tmp/{uuid.uuid4().hex}.png"
    image.save(file_name, format="PNG")
    return file_name

def handler(event: dict) -> dict:
    """
    Main RunPod handler.

    Input JSON:
    {
        "mode": "txt2img" | "img2img",
        "prompt": "..." or ["..."],
        "image_url": "...",        # required for img2img
        "strength": 0.6,           # optional for img2img
        "guidance_scale": 7.5,
        "steps": 30,
        "lora_repo": "user/lora"   # optional
    }

    Returns:
    {
        "image_paths": ["/tmp/image.png", ...]
    }
    """
    mode = event.get("mode", "txt2img")
    prompts = event.get("prompt", "masterpiece, high detail, character design")
    guidance_scale = float(event.get("guidance_scale", 7.5))
    steps = max(5, int(event.get("steps", 30)))
    strength = float(event.get("strength", 0.75))
    lora_repo = event.get("lora_repo")

    if isinstance(prompts, str):
        prompts = [prompts]

    txt2img_pipe, img2img_pipe = load_pipeline_with_lora(MODEL_ID, lora_repo)

    init_image = None
    if mode == "img2img":
        image_url = event.get("image_url")
        if not image_url:
            return {"error": "Missing 'image_url' for img2img mode"}
        init_image = load_image_from_url(image_url)

    default_suffix = ", ultra realistic, 4k, cinematic lighting, soft shadows, intricate detail"
    image_paths = []

    for prompt in prompts:
        enriched_prompt = f"{prompt}{default_suffix}" if "realistic" not in prompt.lower() else prompt

        if mode == "img2img":
            result = img2img_pipe(
                prompt=enriched_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps
            ).images[0]
        else:
            result = txt2img_pipe(
                prompt=enriched_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps
            ).images[0]

        image_paths.append(save_image_to_tmp_and_get_path(result))

    return {"image_paths": image_paths}