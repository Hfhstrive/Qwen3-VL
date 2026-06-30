import glob
import os

import cv2
import torch
import tqdm
import random
from ipdb import set_trace
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


def add_bottom_padding_with_text_pil(image_path, text, output_path, pad_height=20, font_path='/home/inno/code_inno/innoaitools/demo/MicrosoftYH.ttf', font_size=12):
    """
    使用PIL添加padding和文字（支持中文，推荐使用此方法）

    Args:
        image_path: 输入图像路径
        text: 要添加的文字（支持中文）
        output_path: 输出图像路径
        pad_height: padding高度（像素）
        font_path: 字体文件路径（中文需指定中文字体）
        font_size: 字体大小
    """
    img = Image.open(image_path)
    w, h = img.size

    # 创建带padding的新图像（白色背景）
    new_img = Image.new('RGB', (w, h + pad_height), color='white')
    new_img.paste(img, (0, 0))
    # 确保text是字符串
    text = str(text) if text is not None else ""
    # 添加文字
    draw = ImageDraw.Draw(new_img)
    # 设置字体
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            print(f"无法加载字体: {font_path}，使用默认字体")
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    # 计算文字位置（水平居中，垂直居中于padding区域）
    if text:
        # 使用getbbox获取文字边界
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (w - text_width) // 2
        text_y = h + (pad_height - text_height) // 2

        draw.text((text_x, text_y), text, fill='black', font=font)
    new_img.save(output_path)


if __name__ == '__main__':
    vlm='Qwen/Qwen3-VL-2B-Instruct'
    lora_adapter_path = '/home/inno/code/VLM/Qwen3-VL/qwen-vl-finetune/output/V2/lora_qwen3_2b_r64_alpha128_dropout0.05_zero2_448_768_freeze_vision-mlp_lr1e-4/checkpoint-320/'
    # default: Load the model on the available device(s)
    model = AutoModelForImageTextToText.from_pretrained(
        vlm, dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    processor = AutoProcessor.from_pretrained(vlm)

    image_path = '/media/inno/data/gastroscope-det/TestData/ec05_02_20250217_crop/'
    save_path = '/media/inno/output/VLM/Qwen3_VL/2B/ec05_02_20250217_freeze_vision-mlp/'
    os.makedirs(save_path, exist_ok=True)
    images = glob.glob(f'{image_path}/**.jpg', recursive=True)
    for image in tqdm.tqdm(images):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "请根据消化内镜诊治标准，描述图像中的病变及黏膜特征"},
                ],
            }
        ]
        # Preparation for inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # save_txt = os.path.join(save_path, image.split('/')[-1].split('.')[0] + '.txt')
        # with open(save_txt, 'w') as f:
        #     f.writelines(output_text)
        add_bottom_padding_with_text_pil(image, output_text, os.path.join(save_path, image.split('/')[-1]))