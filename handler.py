import os
import torch
import traceback
import warnings
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTokenizer
from runpod.serverless import start

# ──────────────────────────────────────────────────────────────────────────────
#  ENV & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(".env.local", override=True)

# Patch HF environment for stability
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
#  FETCH MODEL (only first run)
# ──────────────────────────────────────────────────────────────────────────────
def download_model_if_needed():
    if not os.path.exists(os.path.join(MODEL_PATH, "model_index.json")):
        print("🔽 First run: downloading Juggernaut-XL …")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_REPO,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16
        )
        pipe.save_pretrained(MODEL_PATH)

        # Save tokenizers explicitly to avoid fallback issues
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

# Restore missing tokenizers (if needed)
if pipe.tokenizer is None:
    print("🩹 tokenizer missing — reloading")
    pipe.tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_REPO, subfolder="tokenizer", use_auth_token=HF_TOKEN
    )
    pipe.tokenizer.save_pretrained(os.path.join(MODEL_PATH, "tokenizer"))

if pipe.tokenizer_2 is None:
    print("🩹 tokenizer_2 missing — reloading")
    pipe.tokenizer_2 = CLIPTokenizer.from_pretrained(
        MODEL_REPO, subfolder="tokenizer_2", use_auth_token=HF_TOKEN
    )
    pipe.tokenizer_2.save_pretrained(os.path.join(MODEL_PATH, "tokenizer_2"))

# Try enabling xformers
try:
    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        print("⚡ xFormers memory-efficient attention enabled.")
except Exception as e:
    print(f"⚠️ Could not enable xFormers: {e}")

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN HANDLER
# ──────────────────────────────────────────────────────────────────────────────
def handler(event):
    try:
        data = event.get("input", {})
        if not isinstance(data, dict):
            raise ValueError("Expected 'input' to be a dictionary")

        prompt   = data.get("prompt", "masterpiece, beautiful girl, cinematic lighting")
        guidance = float(data.get("guidance_scale", 7.5))
        steps    = int(data.get("steps", 30))

        print(f"🎨 Prompt: {prompt!r} | Steps: {steps} | Guidance: {guidance}")

        result = pipe(
            prompt,
            guidance_scale=guidance,
            num_inference_steps=steps
        )

        image = result.images[0]
        out_path = "/tmp/output.png"
        image.save(out_path)

        print("✅ Image generation successful")
        return {"image_paths": [out_path]}

    except Exception as exc:
        print("❌ Generation failed")
        traceback.print_exc()
        return {"error": str(exc), "trace": traceback.format_exc()}

# ──────────────────────────────────────────────────────────────────────────────
#  RUNPOD ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────
start({"handler": handler})