"""项目路径常量。

默认按包位置推断仓库根目录，可用环境变量 TINYVISION_ROOT 覆盖。
所有脚本都不再依赖当前工作目录。
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("TINYVISION_ROOT") or Path(__file__).resolve().parents[2]).resolve()

ASSETS_DIR = ROOT / "assets"
VISION_ENCODER_DIR = ASSETS_DIR / "vision_encoder"
PRETRAINED_DIR = ASSETS_DIR / "pretrained"
OUTPUTS_DIR = ASSETS_DIR / "outputs"
DATA_DIR = ASSETS_DIR / "data"
TOKENIZER_DIR = ASSETS_DIR / "tokenizer"
DEMO_IMAGES_DIR = ASSETS_DIR / "demo_images"

VISION_MODEL_DIR = VISION_ENCODER_DIR / "siglip2-base-p32-256-ve"


def _moe_suffix(use_moe):
    return '_moe' if use_moe else ''


def final_weight_path(prefix, hidden_size, use_moe):
    """训练产出的最终权重：outputs/{prefix}/final/{prefix}_{hidden_size}[_moe].pth"""
    return OUTPUTS_DIR / prefix / 'final' / f'{prefix}_{hidden_size}{_moe_suffix(use_moe)}.pth'


def checkpoint_dir(prefix):
    """训练检查点目录：outputs/{prefix}/checkpoints"""
    return OUTPUTS_DIR / prefix / 'checkpoints'


def resolve_weight_path(prefix, hidden_size, use_moe, save_dir=None):
    """查找加载用权重：先显式 save_dir，再 pretrained/（输入），最后 outputs/{prefix}/final/（训练产物）。"""
    filename = f'{prefix}_{hidden_size}{_moe_suffix(use_moe)}.pth'
    if save_dir:
        return str(Path(save_dir) / filename)
    for candidate in (PRETRAINED_DIR / filename, final_weight_path(prefix, hidden_size, use_moe)):
        if candidate.exists():
            return str(candidate)
    return str(final_weight_path(prefix, hidden_size, use_moe))
