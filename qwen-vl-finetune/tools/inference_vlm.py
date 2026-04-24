import os
import glob
import json
import argparse
import cv2
import uuid
import random
import numpy as np
from ipdb import set_trace
from tqdm import tqdm

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

DEFAULT_VLM = 'Qwen/Qwen3-VL-2B-Instruct'
# default taken from tools/test.py
DEFAULT_LORA = '/home/inno/code/VLM/Qwen3-VL/qwen-vl-finetune/output/V2/lora_qwen3_2b_r64_alpha128_dropout0.05_zero2_448_768_freeze_vision-mlp_lr1e-4/checkpoint-320/'


def crop_invalid_region(img, padding=[0, 0, 0, 0], padding_dynamic=True, DEBUG=False, ignore_square=False):
    # padding parameters: [x_left_padding, x_right_padding, y_top_padding, y_bottom_padding]
    assert min(padding) >= 0 and max(padding) <= 1

    if ignore_square and 0.8 <= img.shape[0] / img.shape[1] <= 1.2:
        return img, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_size = min(img.shape[0], img.shape[1]) // 100
    if (blur_size % 2) == 0:
        blur_size += 1
    gray_blur = cv2.medianBlur(gray, blur_size)

    pixel_gap = 5
    t1 = abs(img[:, :, 0].astype('int32') - img[:, :, 1].astype('int32'))
    t2 = abs(img[:, :, 1].astype('int32') - img[:, :, 2].astype('int32'))
    t3 = abs(img[:, :, 0].astype('int32') - img[:, :, 2].astype('int32'))
    t1[t1 < pixel_gap] = 0
    t1[t1 >= pixel_gap] = 1
    t2[t2 < pixel_gap] = 0
    t2[t2 >= pixel_gap] = 1
    t3[t3 < pixel_gap] = 0
    t3[t3 >= pixel_gap] = 1
    gray_mask = (t1 + t2 + t3).astype('uint8')
    gray_mask[gray_mask > 0] = 1
    element_size = min(img.shape[0], img.shape[1]) // 200
    if (element_size % 2) == 0:
        element_size += 1
    element_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (element_size, element_size))
    gray_mask = cv2.erode(cv2.dilate(gray_mask, element_struct), element_struct)  # 腐蚀

    # original img is gray image
    if gray_mask.max() == 0:
        gray_mask = np.ones(gray_blur.shape).astype('uint8')

    enhance_blur = gray_blur.copy()
    if gray_mask.mean() < 0.6:
        enhance_blur = np.multiply(gray_blur, gray_mask + 0.5)
        enhance_blur[enhance_blur > 255] = 255
        enhance_blur[enhance_blur < 20] = 0
        enhance_blur = enhance_blur.astype('uint8')

    threshold = 30.0
    if enhance_blur.min() > threshold / 3:
        threshold *= enhance_blur.min() / 10
    if enhance_blur.mean() > threshold * 3:
        threshold *= enhance_blur.mean() * 0.75
    if enhance_blur.max() < threshold * 3:
        threshold *= enhance_blur.max() / 255
    if enhance_blur.mean() < threshold:
        threshold = enhance_blur.mean() * 0.5

    (_, mask) = cv2.threshold(enhance_blur, threshold, 255.0, cv2.THRESH_BINARY)

    (contours_ori, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours_ori, key=lambda contour: len(contour), reverse=True)

    satisfy = False
    for contour in contours:
        roi = cv2.boundingRect(contour)
        if 0.7 < roi[2] / roi[3] < 1.5 and min(roi[2], roi[3]) > (min(img.shape[0], img.shape[1]) / 2):
            satisfy = True
            break

    if not satisfy:
        if DEBUG:
            print('max: {}   min: {}   mean: {}   threshold: {}'.format(enhance_blur.max(), enhance_blur.min(), enhance_blur.mean(), threshold))
            cv2.imshow('img', img)
            cv2.imshow('gray_blur', gray_blur)
            cv2.imshow('gray_mask', gray_mask * 255)
            cv2.imshow('enhance_blur', enhance_blur)
            cv2.imshow('mask', mask)
            # cv2.imshow('stitched', stitched)
            cv2.waitKey(0)
        return img, None

    stitched = img.copy()
    if max(padding) > 0:
        weight = roi[2]
        height = roi[3]
        if padding_dynamic:
            padding_factor = [random.uniform(0, padding[0]), random.uniform(0, padding[1]), random.uniform(0, padding[2]), random.uniform(0, padding[3])]
        else:
            padding_factor = padding
        x_left_padding = weight * padding_factor[0]
        x_right_padding = weight * padding_factor[1]
        y_top_padding = height * padding_factor[2]
        y_bottom_padding = height * padding_factor[3]
        roi_x1, roi_y1, roi_w, roi_h = list(roi)
        roi_x2 = roi_x1 + roi_w
        roi_y2 = roi_y1 + roi_h

        new_x1 = 0 if (roi_x1 - x_left_padding) < 0 else int(roi_x1 - x_left_padding)
        new_x2 = stitched.shape[1] if (roi_x2 + x_right_padding) > stitched.shape[1] else int(roi_x2 + x_right_padding)
        new_y1 = 0 if (roi_y1 - y_top_padding) < 0 else int(roi_y1 - y_top_padding)
        new_y2 = stitched.shape[0] if (roi_y2 + y_bottom_padding) > stitched.shape[0] else int(roi_y2 + y_bottom_padding)
        # print('{} {} {} {}'.format(roi_x1, roi_x2, roi_y1, roi_y2))
        # print('{} {} {} {}'.format(new_x1, new_x2, new_y1, new_y2))
        roi = tuple([new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1])
    stitched = stitched[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    return stitched, roi


def main(args):
    images_root = args.input
    out_path = args.output

    os.makedirs(out_path, exist_ok=True)

    # load model + processor
    print('Loading model...')
    model = AutoModelForImageTextToText.from_pretrained(args.vlm, dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, args.lora)
    processor = AutoProcessor.from_pretrained(args.vlm)

    # collect image folders under images_root that end with images_crop
    all_images = glob.glob(f'{images_root}/**/images_crop/*', recursive=True)
    support_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = [p for p in all_images if os.path.splitext(p)[1].lower() in support_ext]

    # group by parent folder
    folders = {}
    for img in images:
        folder = os.path.dirname(img)
        folders.setdefault(folder, []).append(img)

    out_file = os.path.join(out_path, 'vlm_describe.json')

    mapping = {}
    if args.crop_ai:
        tmp_crop_dir = os.path.join(out_path, 'images_crops')
        os.makedirs(tmp_crop_dir, exist_ok=True)

    for folder, imgs in tqdm(folders.items()):
        for image in imgs:
            try:
                image_for_model = image
                if args.crop_ai:
                    try:
                        img = cv2.imread(image)
                        cropped, roi = crop_invalid_region(img, padding=[0.03, 0.03, 0.05, 0.05], padding_dynamic=True,
                                                           ignore_square=False)
                        if roi is not None:
                            tmp_path = os.path.join(tmp_crop_dir, os.path.basename(image))
                            cv2.imwrite(tmp_path, cropped)
                            image_for_model = tmp_path
                    except Exception:
                        image_for_model = image

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_for_model},
                            {"type": "text", "text": "请根据消化内镜诊治标准，简洁地用一句话描述图像中的病变及黏膜特征。"},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(model.device)
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                model_desc = output_texts[0].strip() if output_texts else ''
            except Exception:
                model_desc = ''

            # use basename as the image_name key
            image_name = os.path.basename(image)
            mapping[image_name] = model_desc

    # write final mapping as a single JSON object
    with open(out_file, 'w', encoding='utf-8') as out_f:
        json.dump(mapping, out_f, ensure_ascii=False, indent=4)

    print('Done. Output written to', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/media/inno/VLM/D1_images_and_reports_which_have_video_20250316/胃镜/', help='根目录，寻找 **/images/*')
    parser.add_argument('--output', type=str, default='media/inno/output/VLM/D1_images_and_reports_which_have_video_20250316/胃镜/', help='输出目录')
    parser.add_argument('--vlm', type=str, default=DEFAULT_VLM, help='视觉语言模型名或路径')
    parser.add_argument('--lora', type=str, default=DEFAULT_LORA, help='lora adapter 路径')
    parser.add_argument('--crop_ai', action='store_true')
    args = parser.parse_args()
    main(args)
