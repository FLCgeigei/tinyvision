"""命令行图片问答入口。"""
import argparse
import os
import random
import time
import warnings

import torch
from PIL import Image
from transformers import TextStreamer

from ..inference.engine import load_model
from ..models.vlm import TinyVisionVLM
from ..paths import DEMO_IMAGES_DIR
from ..training.utils import setup_seed

warnings.filterwarnings('ignore')


def build_parser():
    parser = argparse.ArgumentParser(description="TinyVision Chat")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default=None, type=str, help="权重目录（默认在 pretrained/ 与 outputs/*/final/ 中自动查找）")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀（pretrain_vlm, sft_vlm）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--image_dir', default=None, type=str, help="测试图像目录（默认 assets/demo_images）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model, tokenizer, preprocess = load_model(
        load_from=args.load_from, save_dir=args.save_dir, weight=args.weight,
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe, device=args.device,
    )
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    image_dir = args.image_dir or str(DEMO_IMAGES_DIR)
    # 自动测试image_dir中的所有图像
    prompt = "<image>\n请描述这张图中的主要物体和场景。"
    # prompt = "<image>\nPlease illustrate the image through your words."
    for image_file in sorted(os.listdir(image_dir)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            setup_seed(random.randint(1, 31415926))
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            pixel_values = {k: v.to(args.device) for k, v in TinyVisionVLM.image2tensor(image, preprocess).items()}

            messages = [{"role": "user", "content": prompt.replace('<image>', model.config.image_special_token * model.config.image_token_len)}]
            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
            inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)

            print(f'[图像]: {image_file}')
            print(f"💬: {repr(prompt)}")
            print('🤖: ', end='')
            st = time.time()
            generated_ids = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature, pixel_values=pixel_values
            )
            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')


if __name__ == "__main__":
    main()
