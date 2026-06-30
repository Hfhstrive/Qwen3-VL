import glob
import os
import json
import numpy as np
import pandas as pd
import random
import cv2
random.seed(20260416)
from ipdb import set_trace


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            # DataFrame转换为字典列表
            return obj.to_dict(orient='records')
        return json.JSONEncoder.default(self, obj)


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


if __name__ == '__main__':
    crop_ai = True

    ori_paths = [
        '/media/inno/VLM/D2_images_and_reports_20260402/Qwen3-VL/病例/description/',
        '/media/inno/VLM/D2_images_and_reports_20260402/Qwen3-VL/挑选病变/description/'
        ]
    save_path = '/media/inno/VLM/D2_images_and_reports_20260402/Qwen3-VL/datasets/V3/'
    os.makedirs(save_path, exist_ok=True)
    all_json = os.path.join(save_path, 'all.json')
    # train_json = os.path.join(save_path, 'train.json')
    # val_json = os.path.join(save_path, 'val.json')
    info_paths = []
    for ori_path in ori_paths:
        info_paths.extend(glob.glob(f'{ori_path}/**/info.txt', recursive=True))
    count = 0
    datasets = []
    # datasets_train = []
    # datasets_val = []
    for info_path in info_paths:
        with open(info_path, 'r') as f:
            des_infos = f.readlines()
        for des_info in des_infos:
            count += 1
            rand = random.random()
            # datasets = datasets_train if rand < 0.8 else datasets_val
            name, loc, des = des_info.strip('\n').split(' ')[0], des_info.strip('\n').split(' ')[1], des_info.strip('\n').split(' ')[2]
            if '病例' in info_path:
                image_dir = info_path.replace('description', 'ori').replace('info.txt', 'images')
            else:
                image_dir = info_path.replace('description', 'ori').replace('info.txt', '')
            image_path = os.path.join(image_dir, name)
            assert os.path.exists(image_path)
            imagefile = image_path
            if crop_ai:
                img = cv2.imread(image_path)
                valid_region, roi = crop_invalid_region(img, padding=[0.03, 0.03, 0.05, 0.05], padding_dynamic=True,
                                                        ignore_square=False)
                crop_dir = image_dir.replace('ori', 'crop')
                os.makedirs(crop_dir, exist_ok=True)
                imagefile = os.path.join(crop_dir, name)
                if not os.path.exists(imagefile):
                    cv2.imwrite(imagefile, valid_region)
            # 部位
            keywords = ['食管', '胃', '十二指肠']
            loc_coarse = next((k for k in keywords if k in loc), loc)
            # 光源
            light = 'NBI' if 'NBI' in des else '白光'
            # 描述
            des = des.replace('NBI', '')

            image_des = {
                "image": imagefile,
                # "conversations": [
                #     {
                #         "from": "human",
                #         "value": "请根据消化内镜诊治标准，描述图像中的黏膜特征\n<image>"
                #     },
                #     {
                #         "from": "gpt",
                #         # "value": f'当前部位为{loc_coarse}，{des}'   # V1
                #         "value": f'当前部位为{loc_coarse}。光源为{light}。{des}' # V2
                #     }
                # ]
                # V3
                "conversations": [
                    {
                        "from": "human",
                        "value": "观察的消化道部位为？\n<image>"
                    },
                    {
                        "from": "gpt",
                        "value": f'{loc_coarse}'
                    },
                    {
                        "from": "human",
                        "value": "根据消化内镜诊治标准，描述图像中的病变及黏膜特征"
                    },
                    {
                        "from": "gpt",
                        "value": f'{des}'
                    },
                ]
            }
            datasets.append(image_des)
    print(len(datasets))
    # print(len(datasets_train))
    # print(len(datasets_val))
    with open(all_json, 'w', encoding='utf-8') as file:
        json.dump(datasets, file, cls=CustomEncoder, ensure_ascii=False, indent=4)
    # with open(train_json, 'w', encoding='utf-8') as file:
    #     json.dump(datasets_train, file, cls=CustomEncoder, ensure_ascii=False, indent=4)
    # with open(val_json, 'w', encoding='utf-8') as file:
    #     json.dump(datasets_val, file, cls=CustomEncoder, ensure_ascii=False, indent=4)