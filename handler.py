import os
import torch
import traceback
import warnings
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTokenizer
import uvicorn

# ──────────────────────────────────────────────────────────────────────────────
#  ENV SETUP
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(".env.local", override=True)

# Force Hugging Face to cache in mounted volume
os.environ["HF_HOME"] = "/runpod-volume/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/huggingface"
os.environ["HF_HUB_CACHE"] = "/runpod-volume/huggingface"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

HF_TOKEN   = os.getenv("HUGGING_FACE_TOKEN")
MODEL_REPO = "RunDiffusion/Juggernaut-XL-v8"
MODEL_PATH = "/runpod-volume/juggernaut-xl"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
os.makedirs(MODEL_PATH, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  CLEAN BROKEN SNAPSHOTS IF NEEDED
# ──────────────────────────────────────────────────────────────────────────────
SNAPSHOT_ROOT = os.path.join(os.environ["HF_HOME"], "models--RunDiffusion--Juggernaut-XL-v8")
if os.path.exists(SNAPSHOT_ROOT):
    print("🧹 Removing potential broken HF cache…")
    shutil.rmtree(SNAPSHOT_ROOT, ignore_errors=True)

# ──────────────────────────────────────────────────────────────────────────────
#  DOWNLOAD MODEL IF NOT CACHED
# ──────────────────────────────────────────────────────────────────────────────
def download_model_if_needed():
    model_index = os.path.join(MODEL_PATH, "model_index.json")
    if not os.path.exists(model_index):
        print("🔽 First run: downloading Juggernaut-XL …")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_REPO,
            torch_dtype=torch.float16,
            cache_dir=os.environ["HF_HOME"]
        )
        pipe.save_pretrained(MODEL_PATH)
        pipe.tokenizer.save_pretrained(os.path.join(MODEL_PATH, "tokenizer"))
        pipe.tokenizer_2.save_pretrained(os.path.join(MODEL_PATH, "tokenizer_2"))
        del pipe
    else:
        print("✅ Model already cached")

download_model_if_needed()

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to(DEVICE)

# Restore missing tokenizers if needed
if pipe.tokenizer is None:
    print("🩹 tokenizer missing — restoring")
    pipe.tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_REPO, subfolder="tokenizer", use_auth_token=HF_TOKEN
    )
    pipe.tokenizer.save_pretrained(os.path.join(MODEL_PATH, "tokenizer"))

if pipe.tokenizer_2 is None:
    print("🩹 tokenizer_2 missing — restoring")
    pipe.tokenizer_2 = CLIPTokenizer.from_pretrained(
        MODEL_REPO, subfolder="tokenizer_2", use_auth_token=HF_TOKEN
    )
    pipe.tokenizer_2.save_pretrained(os.path.join(MODEL_PATH, "tokenizer_2"))

# Enable memory-efficient attention
try:
    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        print("⚡ xFormers memory-efficient attention enabled.")
except Exception as e:
    print(f"⚠️ Could not enable xFormers: {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  FASTAPI APP
# ──────────────────────────────────────────────────────────────────────────────
class GenerationInput(BaseModel):
    prompt: str = "masterpiece, beautiful girl, cinematic lighting"
    guidance_scale: float = 7.5
    steps: int = 30

app = FastAPI()

@app.post("/generate")
async def generate_image(data: GenerationInput):
    try:
        print(f"🎨 Prompt: {data.prompt!r} | Steps: {data.steps} | Guidance: {data.guidance_scale}")

        result = pipe(
            data.prompt,
            guidance_scale=data.guidance_scale,
            num_inference_steps=data.steps
        )

        image = result.images[0]
        out_path = "/tmp/output.png"
        image.save(out_path, format="PNG")

        print("✅ Image generation successful")
        return {"image_url": "/output.png"}

    except Exception as exc:
        print("❌ Generation failed")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "error": str(exc),
            "trace": traceback.format_exc()
        })

@app.get("/output.png")
async def get_output_image():
    image_path = "/tmp/output.png"
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "Image not found"})

if __name__ == "__main__":
    uvicorn.run("handler:app", host="0.0.0.0", port=3000, reload=True)