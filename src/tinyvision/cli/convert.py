import argparse
import os
import json

import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from ..models.vlm import TinyVisionVLM, VLMConfig
from ..paths import OUTPUTS_DIR, TOKENIZER_DIR, VISION_MODEL_DIR, final_weight_path

warnings.filterwarnings('ignore', category=UserWarning)


def convert_to_transformers(torch_path, transformers_path, lm_config, dtype=torch.bfloat16):
    VLMConfig.register_for_auto_class()
    TinyVisionVLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = TinyVisionVLM(lm_config, vision_model_path=str(VISION_MODEL_DIR))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    del lm_model.vision_encoder
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.save_pretrained(transformers_path)
    # 显式写入 tie_word_embeddings（save_pretrained 默认不序列化与默认值相同的字段）
    config_path = os.path.join(transformers_path, "config.json")
    config = json.load(open(config_path, 'r', encoding='utf-8'))
    config['tie_word_embeddings'] = True
    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; config.pop('rope_parameters', None)
    json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-TinyVision 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert TinyVision weights to Transformers format")
    parser.add_argument('--torch_path', default=None, type=str, help="原生 pth 权重路径（默认 assets/outputs/sft_vlm/final/sft_vlm_*.pth）")
    parser.add_argument('--transformers_path', default=None, type=str, help="Transformers 输出目录（默认 assets/outputs/transformers/tinyvision-3v*）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=1, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    args = parser.parse_args(argv)

    lm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                          max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe))
    suffix = '_moe' if args.use_moe else ''
    torch_path = args.torch_path or str(final_weight_path('sft_vlm', lm_config.hidden_size, bool(args.use_moe)))
    transformers_path = args.transformers_path or str(OUTPUTS_DIR / 'transformers' / f'tinyvision-3v{suffix}')
    convert_to_transformers(torch_path, transformers_path, lm_config)


if __name__ == '__main__':
    main()
