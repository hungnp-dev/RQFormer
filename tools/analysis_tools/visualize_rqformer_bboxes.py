# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize a small image sample with predicted rotated bounding boxes."""

import argparse
import os
import os.path as osp
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import DictAction

from mmrotate.structures.bbox import QuadriBoxes, RotatedBoxes
from mmrotate.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize RQFormer predicted rotated boxes')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('img', help='image file or image directory')
    parser.add_argument('--out-dir', default='work_dirs/bbox_visualizations')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument(
        '--max-images',
        type=int,
        default=10,
        help='0 means visualize all images when img is a directory')
    parser.add_argument(
        '--group-by-class',
        action='store_true',
        help='select and save bbox images by ground-truth class when annotations are available')
    parser.add_argument(
        '--images-per-class',
        type=int,
        default=2,
        help='number of images per class when --group-by-class is enabled')
    parser.add_argument(
        '--candidates-per-class',
        type=int,
        default=12,
        help='candidate images scored per class before keeping the best samples')
    parser.add_argument(
        '--target-classes',
        nargs='*',
        default=None,
        help='optional class subset for --group-by-class; defaults to all dataset classes')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options, e.g. key=value')
    return parser.parse_args()


def collect_images(path):
    path = Path(path)
    if path.is_file():
        return [str(path)]

    suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    return sorted(
        str(p) for p in path.rglob('*') if p.suffix.lower() in suffixes)


def text_origin(poly, image_shape):
    x = int(max(2, np.min(poly[:, 0])))
    y = int(max(14, np.min(poly[:, 1]) - 4))
    h, w = image_shape[:2]
    return min(x, max(2, w - 80)), min(y, max(14, h - 4))


def draw_label(image_bgr, text, origin, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(image_bgr, (x, y - th - baseline - 3),
                  (x + tw + 4, y + baseline), color, -1)
    cv2.putText(image_bgr, text, (x + 2, y - 2), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_polygons(image_bgr, polygons, labels, class_names, color, prefix):
    for poly, label in zip(polygons, labels):
        poly = np.asarray(poly, dtype=np.float32).reshape(4, 2)
        cls = class_names[int(label)] if class_names and int(label) < len(class_names) else str(label)
        cv2.polylines(image_bgr, [poly.astype('int32')], True, color, 2)
        draw_label(image_bgr, f'{prefix}:{cls}', text_origin(poly, image_bgr.shape), color)


def draw_rotated_boxes(image_bgr, pred_instances, score_thr, class_names=None):
    if pred_instances is None or len(pred_instances) == 0:
        return image_bgr

    scores = pred_instances.scores.detach().cpu()
    keep = scores >= score_thr
    if not keep.any():
        return image_bgr

    bboxes = pred_instances.bboxes[keep]
    labels = pred_instances.labels[keep].detach().cpu().numpy()
    kept_scores = scores[keep].numpy()

    if isinstance(bboxes, torch.Tensor):
        if bboxes.size(-1) == 5:
            bboxes = RotatedBoxes(bboxes)
        elif bboxes.size(-1) == 8:
            bboxes = QuadriBoxes(bboxes)
        else:
            return image_bgr

    polys = bboxes.cpu().convert_to('qbox').tensor.numpy().reshape(-1, 4, 2)
    palette = [(255, 0, 255), (0, 180, 255), (0, 210, 0), (255, 128, 0),
               (255, 0, 0), (0, 255, 255), (180, 60, 255)]
    for poly, label, score in zip(polys, labels, kept_scores):
        color = palette[int(label) % len(palette)]
        cls = class_names[int(label)] if class_names and int(label) < len(class_names) else str(label)
        cv2.polylines(image_bgr, [poly.astype('int32')], True, color, 2)
        draw_label(image_bgr, f'{cls} {score:.2f}', text_origin(poly, image_bgr.shape), color)
    return image_bgr


def dior_gt_path(img_path):
    path = Path(img_path)
    parts = list(path.parts)
    if 'JPEGImages-test' in parts:
        idx = parts.index('JPEGImages-test')
        root = Path(*parts[:idx])
        return root / 'Annotations' / 'Oriented Bounding Boxes' / f'{path.stem}.xml'
    return None


def load_dior_gt(img_path, class_names):
    xml_path = dior_gt_path(img_path)
    if not xml_path or not xml_path.exists():
        return [], []
    cat_to_label = {name: i for i, name in enumerate(class_names or [])}
    polygons, labels = [], []
    root = ET.parse(xml_path).getroot()
    for obj in root.findall('object'):
        name = obj.findtext('name', default='').lower()
        box = obj.find('robndbox')
        if box is None or name not in cat_to_label:
            continue
        values = [
            box.findtext('x_left_top'), box.findtext('y_left_top'),
            box.findtext('x_right_top'), box.findtext('y_right_top'),
            box.findtext('x_right_bottom'), box.findtext('y_right_bottom'),
            box.findtext('x_left_bottom'), box.findtext('y_left_bottom'),
        ]
        try:
            polygons.append(np.array([float(v) for v in values], dtype=np.float32).reshape(4, 2))
            labels.append(cat_to_label[name])
        except (TypeError, ValueError):
            continue
    return polygons, labels


def draw_ground_truth(image_bgr, img_path, class_names):
    polygons, labels = load_dior_gt(img_path, class_names)
    if polygons:
        draw_polygons(image_bgr, polygons, labels, class_names, (80, 220, 80), 'GT')
    return image_bgr



DIOR_CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')


def gt_class_names(img_path):
    xml_path = dior_gt_path(img_path)
    if not xml_path or not xml_path.exists():
        return []
    names = []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    for obj in root.findall('object'):
        name = obj.findtext('name', default='').lower()
        if name and name not in names:
            names.append(name)
    return names


def safe_class_dir(name):
    return ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in name)


def class_score(pred_instances, class_id):
    if pred_instances is None or len(pred_instances) == 0:
        return 0.0, 0
    labels = pred_instances.labels.detach().cpu()
    scores = pred_instances.scores.detach().cpu()
    keep = labels == int(class_id)
    if not keep.any():
        return 0.0, 0
    return float(scores[keep].max().item()), int(keep.sum().item())


def collect_class_candidates(img_paths, class_names, per_class, target_classes=None):
    class_names = list(class_names or DIOR_CLASSES)
    targets = [c.lower() for c in (target_classes or class_names)]
    candidates = {name: [] for name in targets}

    for img_path in img_paths:
        present = set(gt_class_names(img_path))
        for name in targets:
            if len(candidates[name]) < per_class and name in present:
                candidates[name].append(img_path)
        if all(len(paths) >= per_class for paths in candidates.values()):
            break
    return candidates


def grouped_image_tasks(model, img_paths, out_dir, class_names, images_per_class,
                        candidates_per_class, score_thr, target_classes=None):
    class_names = list(class_names or DIOR_CLASSES)
    cat_to_label = {name.lower(): i for i, name in enumerate(class_names)}
    candidates = collect_class_candidates(
        img_paths, class_names, candidates_per_class, target_classes=target_classes)

    result_cache = {}
    ranked = {name: [] for name in candidates}
    for name, paths in candidates.items():
        class_id = cat_to_label.get(name)
        if class_id is None:
            continue
        for img_path in paths:
            if img_path not in result_cache:
                result_cache[img_path] = inference_detector(model, img_path)
            score, det_count = class_score(result_cache[img_path].pred_instances, class_id)
            ranked[name].append((score, det_count, img_path))
        ranked[name].sort(key=lambda item: (item[0], item[1]), reverse=True)

    tasks = []
    selected = {}
    for name, items in ranked.items():
        class_dir = Path(out_dir) / 'by_class' / safe_class_dir(name)
        qualified = [item for item in items if item[0] >= score_thr and item[1] > 0]
        best = qualified[:images_per_class]
        if len(best) < images_per_class:
            used = {img_path for _, _, img_path in best}
            fallback = [item for item in items if item[2] not in used]
            best += fallback[:images_per_class - len(best)]
        selected[name] = [(img_path, score) for score, _, img_path in best]
        for score, _, img_path in best:
            tasks.append((img_path, name, class_dir, result_cache[img_path], score))
    return tasks, selected

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    model = init_detector(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options)

    img_paths = collect_images(args.img)
    class_names = model.dataset_meta.get('classes', None) if hasattr(model, 'dataset_meta') else None

    if args.group_by_class:
        by_class_dir = Path(args.out_dir) / 'by_class'
        if by_class_dir.exists():
            shutil.rmtree(by_class_dir)
        tasks, selected = grouped_image_tasks(
            model, img_paths, args.out_dir, class_names, args.images_per_class,
            args.candidates_per_class, args.score_thr, target_classes=args.target_classes)
        summary = ', '.join(
            f'{name}:{len(items)} best={max((score for _, score in items), default=0):.2f}'
            for name, items in selected.items())
        weak = [name for name, items in selected.items()
                if len(items) < args.images_per_class
                or max((score for _, score in items), default=0) < args.score_thr]
        if weak:
            print('[BBOX WARNING] weak classes:', ', '.join(weak))
        print(f'[BBOX BY CLASS] {summary}')
    else:
        if args.max_images > 0:
            img_paths = img_paths[:args.max_images]
        tasks = [(img_path, None, Path(args.out_dir), None, None) for img_path in img_paths]

    saved = 0
    for img_path, class_name, out_dir, result, score in tasks:
        if result is None:
            result = inference_detector(model, img_path)
        image = mmcv.imread(img_path)
        image = draw_ground_truth(image, img_path, class_names)
        image = draw_rotated_boxes(image, result.pred_instances,
                                   args.score_thr, class_names=class_names)

        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f'_{score:.2f}' if score is not None else ''
        out_name = osp.splitext(osp.basename(img_path))[0] + suffix + '_bbox.jpg'
        out_path = out_dir / out_name
        mmcv.imwrite(image, str(out_path))
        saved += 1
    print(f'[BBOX SAVED] {saved} images -> {args.out_dir}')


if __name__ == '__main__':
    main()
