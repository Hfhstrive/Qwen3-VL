import glob
import json
import os
from ipdb import set_trace
from PIL import Image
import re
from collections import defaultdict
import argparse


def average_hash(path, hash_size=8):
    try:
        img = Image.open(path).convert('L').resize((hash_size, hash_size), Image.BILINEAR)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        return int(bits, 2)
    except Exception:
        return None


def hamming(a, b):
    if a is None or b is None:
        return 64
    x = a ^ b
    return x.bit_count()


# 分组函数
def locate_group(loc_text):
    if not loc_text:
        return '无效'
    for key in ['食管', '胃体', '贲门', '胃角', '胃底']:
        if key in loc_text:
            return key
    if '胃窦' in loc_text or '幽门' in loc_text:
        return '胃窦'
    if '十二指肠' in loc_text:
        return '十二指肠'
    if loc_text == '胃':
        return '胃体'
    return '无效'


def select_representatives(img_dir, filenames, case_info, hash_threshold=5, group_images=3):
    """聚类并选择代表图像。

    规则：
      1) 排除质量不为 '正常质量' 或 体内外不为 '体内' 的图片
      2) 根据部位将图像分为 8 组：食管、胃体、贲门、胃角、胃底、胃窦（幽门）、十二指肠、无效
      3) 每组通过哈希去重并最多保留 3 张，优先选择：
         a) 有 det 描述的图片（det 作为 combined_desc）
         b) det 为空时使用 vlm 描述
         c) 优先保持描述多样性，若描述重复则补充不同分辨率的图片
    """
    # 构建 image -> info 映射，case_info 结构为 {case_no: [ {img: {...}}, ... ]}
    info_map = {}
    if isinstance(case_info, dict):
        for v in case_info.values():
            if isinstance(v, list):
                for entry in v:
                    if isinstance(entry, dict):
                        for name, info in entry.items():
                            info_map[name] = info

    # 收集候选图片并过滤质量/体内外
    items = []
    for fn in filenames:
        path = os.path.join(img_dir, fn)
        name = os.path.basename(path)
        info = info_map.get(name, {})
        quality = info.get('质量', '')
        vitro = info.get('体内外', '')
        if quality != '正常质量' or vitro != '体内':
            continue

        det_text = info['det']
        vlm_text = info['vlm']
        loc = info['位置']
        loc_desc = ''
        if len(loc) > 0:
            loc_desc = f'当前部位为{loc}。'
        combined = loc_desc + det_text if det_text else loc_desc + vlm_text
        items.append({
            'path': path,
            'name': name,
            'hash': average_hash(path),
            'loc': loc,
            'det_present': bool(det_text),
            'combined': combined.strip(),
        })

    if not items:
        return []

    groups = defaultdict(list)
    for it in items:
        grp = locate_group(it['loc'])
        groups[grp].append(it)

    reps_selected = []

    # 处理每个组：基于 hash 聚类，每簇选出最优候选，最后选最多 3 张
    for grp_name in ['食管', '胃体', '贲门', '胃角', '胃底', '胃窦', '十二指肠', '无效']:
        group_items = groups.get(grp_name, [])
        if not group_items:
            continue

        # 基于 hash 做简单聚类
        clusters = []
        for it in group_items:
            placed = False
            for cl in clusters:
                rep_h = cl[0]['hash']
                if it['hash'] is None or rep_h is None:
                    continue
                if hamming(it['hash'], rep_h) <= hash_threshold:
                    cl.append(it)
                    placed = True
                    break
            if not placed:
                clusters.append([it])

        # 每个簇选出最佳候选（优先 det_present，再按 combined 长度）
        cluster_candidates = []
        for cl in clusters:
            best = None
            best_score = (-1, 0)
            for it in cl:
                score = (1 if it['det_present'] else 0, len(it['combined'] or ''))
                if score > best_score:
                    best = it
                    best_score = score
            cluster_candidates.append({'best': best, 'cluster': cl, 'score': best_score})

        # 按优先级排序并尝试挑选最多 group_images 张，保证描述多样性
        cluster_candidates.sort(key=lambda x: (x['score'][0], x['score'][1]), reverse=True)
        selected = []
        seen_desc = set()
        for ci in cluster_candidates:
            if len(selected) >= group_images:
                break
            cand = ci['best']
            desc = (cand['combined'] or '').strip()
            if desc and desc in seen_desc:
                continue
            selected.append(cand)
            if desc:
                seen_desc.add(desc)

        # 若不足 5 张，从各簇中补充不同描述的图片
        if len(selected) < group_images:
            for ci in cluster_candidates:
                if len(selected) >= group_images:
                    break
                for it in ci['cluster']:
                    if it in selected:
                        continue
                    desc = (it['combined'] or '').strip()
                    if desc and desc in seen_desc:
                        continue
                    selected.append(it)
                    if desc:
                        seen_desc.add(desc)
                    break

        # 最后仍不足时，按分辨率补齐
        # if len(selected) < args.group_images:
        #     remaining = [it for cl in clusters for it in cl if it not in selected]
        #     def area(it):
        #         try:
        #             with Image.open(it['path']) as im:
        #                 w, h = im.size
        #                 return w * h
        #         except Exception:
        #             try:
        #                 return os.path.getsize(it['path'])
        #             except Exception:
        #                 return 0
        #     remaining.sort(key=lambda it: area(it), reverse=True)
        #     for it in remaining:
        #         if len(selected) >= args.group_images:
        #             break
        #         selected.append(it)

        reps_selected.extend(selected)

    return reps_selected


def generate_det_description(det):
    # # 按类别分组存储

    # det = [
    #     {'类别': '反流性食管炎', '面积': '0.01', 'LA分级': 'LA-A'},
    #     {'类别': '反流性食管炎', '面积': '0.02', 'LA分级': 'LA-B'},
    #     {'类别': '胃癌', '面积': '0.24'},
    #     {'类别': '胃息肉', '面积': '0.07'},
    #     {'类别': '胃溃疡', '面积': '0.12', 'ahs分期': 'A1', 'Forrest分级': 'Ia'},
    #     {'类别': '胃息肉', '面积': '0.07'},
    #     {'类别': '胃溃疡', '面积': '0.12', 'ahs分期': 'A3', 'Forrest分级': 'IIa'},
    # ]

    category_data = defaultdict(lambda: {'count': 0, 'items': []})

    for item in det:
        category = item['类别']

        # 特殊处理：只记一次，且跳过后续的
        if category in ['食管静脉曲张', '反流性食管炎', '霉菌性食管炎', 'Barrett食管']:
            if category_data[category]['count'] == 0:
                category_data[category]['count'] = 1
                # 收集除了'类别'和'面积'外的其他字段
                extra_keys = [k for k in item.keys() if k not in ['类别', '面积']]
                if extra_keys:
                    extra_dict = {key: item[key] for key in extra_keys}
                    category_data[category]['items'].append(extra_dict)
                else:
                    category_data[category]['items'].append({})
            continue

        category_data[category]['count'] += 1

        # 收集除了'类别'和'面积'外的其他字段
        extra_keys = [k for k in item.keys() if k not in ['类别', '面积']]
        if extra_keys:
            extra_dict = {key: item[key] for key in extra_keys}
            category_data[category]['items'].append(extra_dict)
        else:
            category_data[category]['items'].append({})  # 无额外字段时添加空字典

    # 生成描述
    descriptions = ''
    for category, data in category_data.items():
        count = data['count']
        items = data['items']

        if count == 1:
            # 单个病变，直接输出
            if items[0]:
                extra_parts = [f"{key}为{value}" for key, value in items[0].items()]
                extra_str = '，' + '，'.join(extra_parts)
            else:
                extra_str = ''
            descriptions += f"识别到1处{category}{extra_str}。"
        else:
            # 多个病变，需要分别列出
            item_descs = []
            for idx, extra_dict in enumerate(items, 1):
                if extra_dict:
                    extra_parts = [f"{key}为{value}" for key, value in extra_dict.items()]
                    extra_str = '，'.join(extra_parts)
                    item_descs.append(f"1个病变{extra_str}")

            # 合并描述，用逗号分隔
            if len(item_descs) > 0:
                items_str = '，'.join(item_descs)
                descriptions += f"识别到{count}处{category}，{items_str}。"
            else:
                descriptions += f"识别到{count}处{category}。"

    return descriptions


def merge_det_vlm(img_path, imgs, det_data, vlm_data):
    """合并 det_data 与 vlm_data，返回以病例号/图像名为 key 的字典。

    每个 entry 包含：
      - 图像质量
      - 体内外
      - 部位
      - det_describe： 前置模型的描述
      - vlm_describe： vlm模型的描述
    """
    case_no = img_path.split('/')[-2]
    case_info = {
        case_no: []
    }

    for img in imgs:
        det_entry = det_data.get(img) if isinstance(det_data, dict) else None
        vlm_entry = vlm_data.get(img) if isinstance(vlm_data, dict) else None

        loc = ''
        quality = ''
        vitro = ''
        status_scribe = ''
        det_scribe = ''
        vlm_describe = ''

        det_flag = False
        if det_entry is not None:
            loc = det_entry['位置'] if det_entry['位置'] != '未识别' else ''
            status = det_entry['状态']
            light = '白光' if status['光源'] == 'WLI' else status['光源']
            mag = '放大' if '非放大' not in status['放大'] else '非放大'
            quality = status['质量']
            vitro = status['体内外']
            # 状态描述
            status_scribe = f"在{light}{mag}下观察，"
            if status['染色'] != '无染色':
                status_scribe += f"存在{status['染色']}，"
            if status['手术帽'] != '无手术帽':
                status_scribe += f"{status['手术帽']}，"
            if status['器械'] != '无器械':
                status_scribe += f"{status['器械']}，"
            det_scribe += status_scribe
            # 检测结果
            if len(det_entry['检测']) != 0:
                det_flag = True
                det_scribe += generate_det_description(det_entry['检测'])
            # 当IPCL和萎缩一起出现时，以部位判断
            # assert not (len(det_entry['ipcl']) != 0 and len(det_entry['萎缩']) != 0)
            if len(det_entry['ipcl']) != 0 and len(det_entry['萎缩']) != 0:
                if '食管' in loc:
                    det_flag = True
                    det_scribe += f"IPCL呈{det_entry['ipcl']}型。"
                else:
                    det_flag = True
                    det_scribe += f"背景黏膜存在萎缩。"
            else:
                if len(det_entry['ipcl']) != 0:
                    det_flag = True
                    det_scribe += f"IPCL呈{det_entry['ipcl']}型。"
                if len(det_entry['萎缩']) != 0:
                    det_flag = True
                    det_scribe += f"背景黏膜存在萎缩。"

        if vlm_entry is not None:
            des_idx = 0
            if '部位' in vlm_entry:
                des_idx += 1
                if loc == '':
                    loc = re.search(r'部位为([^。]+)', vlm_entry).group(1)
            if '光源' in vlm_entry:
                des_idx += 1
                if len(status_scribe) == '':
                    light_vlm = re.search(r'光源为([^。]+)', vlm_entry).group(1)
                    status_scribe = f"在{light_vlm}下观察，"

            vlm_describe = status_scribe + '。'.join(vlm_entry.split('。')[des_idx:])

        entry = {
            img: {
                '质量': quality,
                '体内外': vitro,
                '位置': loc,
                'det': det_scribe if det_flag else '',
                'vlm': vlm_describe if vlm_describe.endswith('。') else vlm_describe + '。',
            }
        }
        case_info[case_no].append(entry)

    save_path = os.path.join(args.save_dir, 'merge_describe.jsonl')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(case_info, ensure_ascii=False, indent=4))
    return case_info


def main(args):
    case_path = args.ori_path
    det_path = args.det_path
    vlm_path = args.vlm_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # load detection json referenced for det_summary
    det_data = {}
    if os.path.exists(det_path):
        try:
            with open(det_path, 'r', encoding='utf-8') as f:
                det_data = json.load(f)
        except Exception:
            det_data = {}

    vlm_data = {}
    if os.path.exists(vlm_path):
        try:
            with open(vlm_path, 'r', encoding='utf-8') as f:
                vlm_data = json.load(f)
        except Exception:
            vlm_data = {}

    for id, case in enumerate(sorted(os.listdir(case_path))):
        case_dir = os.path.join(case_path, case)
        report = os.path.join(case_dir, 'info.txt')
        assistant_info = ''
        case_info = ''
        if os.path.exists(report):
            with open(report, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if '检查过程' in line or '检查结果' in line:
                        assistant_info += line.strip() + '\n'

        img_path = os.path.join(case_dir, 'images')
        if not os.path.isdir(img_path):
            continue
        imgs = sorted(os.listdir(img_path))
        if len(imgs) <= args.min_images:
            continue

        merge_info = merge_det_vlm(img_path, imgs, det_data, vlm_data)

        reps = select_representatives(img_path, imgs, merge_info, hash_threshold=args.hash_threshold,
                                      group_images=args.group_images)

        # 按 8 个部位分组输出
        groups_order = ['食管', '胃体', '贲门', '胃角', '胃底', '胃窦', '十二指肠', '无效']
        grouped = {g: [] for g in groups_order}
        for it in reps:
            grp = locate_group(it['loc'])
            if grp not in grouped:
                grouped['无效'].append(it)
            else:
                grouped[grp].append(it)

        grouped_parts = []
        for g in groups_order:
            items_g = grouped.get(g, [])
            if not items_g:
                continue
            part = f"{g}下有{len(items_g)}张图像："
            descs = []
            for idx, it in enumerate(items_g, 1):
                name = it.get('name')
                path = it.get('path')
                combined = it.get('combined').strip()
                descs.append(f"第{idx}张图像:{combined}")
            part += ''.join(descs)
            grouped_parts.append(part)

        case_info = '\n'.join(grouped_parts)

        user_info = f'该上消化道内镜下的病例套图中, 其模型识别的特征及病变如下所示，请帮我生成内镜报告。其模型详细结果如下: {case_info}'
        messages = {"messages": [{"role": "user", "content": user_info},
                                 {"role": "assistant", "content": assistant_info}]}

        status = 'train' if id <= 0.9 * len(os.listdir(case_path)) else 'test'
        save_path = os.path.join(save_dir, status + '.jsonl')
        with open(save_path, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(messages, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str,
                        default='/media/inno/VLM/D1_images_and_reports_which_have_video_20250316/胃镜/',
                        help='原始数据根路径')
    parser.add_argument('--det_path', type=str,
                        default='/media/inno/VLM/D1_images_and_reports_which_have_video_20250316/base/det/胃镜_V3.json',
                        help='检测结果文件')
    parser.add_argument('--vlm_path', type=str,
                        default='/media/inno/VLM/D1_images_and_reports_which_have_video_20250316/base/generated_llm/vlm_describe.json',
                        help='VLM描述文件')
    parser.add_argument('--save_dir', type=str,
                        default='/media/inno/VLM/D1_images_and_reports_which_have_video_20250316/MedicalGPT/V3/',
                        help='输出保存目录')
    parser.add_argument('--hash_threshold', type=int, default=5, help='均值哈希汉明距离阈值')
    parser.add_argument('--group_images', type=int, default=5, help='每个部位最多保留的图像数')
    parser.add_argument('--min_images', type=int, default=5, help='每病例最少图片数，小于则跳过')
    args = parser.parse_args()
    main(args)
