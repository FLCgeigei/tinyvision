"""模型加载：原生 PyTorch 权重或 Transformers 格式。"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models.vlm import TinyVisionVLM, VLMConfig
from ..paths import TOKENIZER_DIR, VISION_MODEL_DIR, resolve_weight_path
from ..training.utils import get_model_params


def load_model(load_from="model", save_dir=None, weight="sft_vlm", hidden_size=768,
               num_hidden_layers=8, use_moe=0, device="cpu"):
    """load_from='model' 时加载原生 pth；其余路径视为 Transformers 格式目录。"""
    if load_from == "model":
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        ckp = resolve_weight_path(weight, hidden_size, bool(use_moe), save_dir)
        model = TinyVisionVLM(
            VLMConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_moe=bool(use_moe)),
            vision_model_path=str(VISION_MODEL_DIR),
        )
        state_dict = torch.load(ckp, map_location=device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_from, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = TinyVisionVLM.get_vision_model(str(VISION_MODEL_DIR))
    get_model_params(model, model.config)
    model = model.eval()
    if "cuda" in device:
        model = model.half()
    return model.to(device), tokenizer, model.processor
