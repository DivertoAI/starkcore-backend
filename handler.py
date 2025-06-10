import os
import torch
import traceback
import warnings
from dotenv import load_dotenv
from diffusers import StableDiffusionXLPipeline
from runpod.serverless import start

# ──────────────────────────────────────────────────────────────────────────────
#  ENV & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv(".env.local", override=True)

HF_TOKEN   = os.getenv("HUGGING_FACE_TOKEN")
MODEL_REPO = "RunDiffusion/Juggernaut-XL-v8"
MODEL_PATH = "/runpod-volume/juggernaut-xl"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
os.makedirs(MODEL_PATH, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  FETCH MODEL (only first run)
# ──────────────────────────────────────────────────────────────────────────────
def download_model_if_needed() -> None:
    if not os.path.exists(os.path.join(MODEL_PATH, "model_index.json")):
        print("🔽  First run: downloading Juggernaut-XL to /runpod-volume …")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_REPO,
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16
        )

        # Core pipeline
        pipe.save_pretrained(MODEL_PATH)

        # Tokenizers **must** live in sub-folders the loader expects
        pipe.tokenizer.save_pretrained(os.path.join(MODEL_PATH, "tokenizer"))
        pipe.tokenizer_2.save_pretrained(os.path.join(MODEL_PATH, "tokenizer_2"))

        # Text-encoders are already saved automatically inside their own sub-folders
        # by pipe.save_pretrained() for diffusers ≥0.23, so no extra calls needed.

        del pipe
    else:
        print("✅  Model already cached at /runpod-volume")

download_model_if_needed()

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL FOR INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to(DEVICE)

try:
    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
        print("⚡  xFormers memory-efficient attention enabled")
except Exception as err:
    print(f"⚠️  Could not enable xFormers: {err}")

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN HANDLER
# ──────────────────────────────────────────────────────────────────────────────
def handler(event):
    """
    Expects JSON like:
    {
      "input": {
         "prompt": "your prompt",
         "guidance_scale": 7.5,
         "steps": 30
      }
    }
    """
    try:
        data = event.get("input", {})
        if not isinstance(data, dict):
            raise ValueError("Input payload must be a dict under key 'input'")

        prompt   = data.get("prompt", "masterpiece, beautiful girl, cinematic lighting")
        guidance = float(data.get("guidance_scale", 7.5))
        steps    = int(data.get("steps", 30))

        print(f"🎨  Prompt: {prompt!r} | Steps: {steps} | Guidance: {guidance}")

        image = pipe(
            prompt,
            guidance_scale=guidance,
            num_inference_steps=steps
        ).images[0]

        out_path = "/tmp/output.png"
        image.save(out_path)
        print("✅  Image generation successful")

        return {"image_paths": [out_path]}

    except Exception as exc:
        print("❌  Generation failed")
        traceback.print_exc()
        return {
            "error": str(exc),
            "trace": traceback.format_exc()
        }

# ──────────────────────────────────────────────────────────────────────────────
#  START SERVERLESS HANDLER
# ──────────────────────────────────────────────────────────────────────────────
start({"handler": handler})