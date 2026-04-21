import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

lora_adapter_path = '//home/inno/code/VLM/Qwen3-VL/qwen-vl-finetune/output/V2/lora_qwen3_2b_r64_alpha128_dropout0.05_zero2_448_768_tune_mm_vision_lr5e-4/checkpoint-320/'
# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(
    # "Qwen/Qwen3-VL-235B-A22B-Instruct", dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_adapter_path)
# processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/media/inno/data/EC05/EC05-02/TrainData/DET04-D2/DET04-Train-D1/train/c709e14e1e1311ea9fe0000c29e37e62.jpg",
                # "image": "/media/inno/VLM/D2_images_and_reports_20260402/Qwen3-VL/挑选病变/crop/1_eca/11c1044af5c711f08576305a3a77b88e.jpg",
                # "image": "/media/inno/VLM/D2_images_and_reports_20260402/Qwen3-VL/挑选病变/crop/1_eca/11c2901ef5c711f08576305a3a77b88e.jpg",
            },
            # {"type": "text", "text": "观察的位置为？"},
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
print(output_text)
