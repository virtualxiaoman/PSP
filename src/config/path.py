from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_DIR = ASSETS_DIR / "model"
MODEL_DINOV2_DIR = MODEL_DIR / "dinov2"
MODEL_DISTILL_DIR = MODEL_DIR / "distill"
MODEL_VIT_DIR = MODEL_DIR / "vit"

# print(f"项目根目录: {PROJECT_ROOT}")
# print(f"资源目录: {ASSETS_DIR}")
# print(MODEL_DINOV2_DIR)

