import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(
    # "Qwen/Qwen3-VL-235B-A22B-Instruct", dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)
# processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/media/inno/data/EC05/EC05-02/TrainData/DET04-D2/DET04-Train-D1/train/c709e14e1e1311ea9fe0000c29e37e62.jpg",
            },
            {"type": "text", "text": "请根据<消化内镜诊治标准术语集(2020)>，用一句话描述黏膜特征."},
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
