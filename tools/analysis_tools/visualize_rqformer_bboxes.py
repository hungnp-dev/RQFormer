# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize a small image sample with predicted rotated bounding boxes."""

import argparse
import os
import os.path as osp
from pathlib import Path

import cv2
import mmcv
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


def draw_rotated_boxes(image_bgr, pred_instances, score_thr):
    if pred_instances is None or len(pred_instances) == 0:
        return image_bgr

    scores = pred_instances.scores.detach().cpu()
    keep = scores >= score_thr
    if not keep.any():
        return image_bgr

    bboxes = pred_instances.bboxes[keep]
    labels = pred_instances.labels[keep].detach().cpu().numpy()

    if isinstance(bboxes, torch.Tensor):
        if bboxes.size(-1) == 5:
            bboxes = RotatedBoxes(bboxes)
        elif bboxes.size(-1) == 8:
            bboxes = QuadriBoxes(bboxes)
        else:
            return image_bgr

    polys = bboxes.cpu().convert_to('qbox').tensor.numpy().reshape(-1, 4, 2)
    palette = [(255, 0, 255), (0, 255, 0), (0, 255, 255), (255, 128, 0),
               (255, 0, 0), (0, 255, 255)]
    for poly, label in zip(polys, labels):
        color = palette[int(label) % len(palette)]
        cv2.polylines(image_bgr, [poly.astype('int32')], True, color, 2)
    return image_bgr


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
    if args.max_images > 0:
        img_paths = img_paths[:args.max_images]

    for img_path in img_paths:
        result = inference_detector(model, img_path)
        image = mmcv.imread(img_path)
        image = draw_rotated_boxes(image, result.pred_instances,
                                   args.score_thr)

        out_name = osp.splitext(osp.basename(img_path))[0] + '_bbox.jpg'
        out_path = osp.join(args.out_dir, out_name)
        mmcv.imwrite(image, out_path)
        print(f'[SAVED] {out_path}')


if __name__ == '__main__':
    main()
