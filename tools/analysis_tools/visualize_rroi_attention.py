# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize RRoI attention as image heatmaps.

This tool is intentionally lightweight: it hooks the existing RRoIAttention
module at inference time, averages its 7x7 attention weights, upsamples the
map to image size, overlays it on the original image, and draws predicted
rotated boxes.
"""

import argparse
import os
import os.path as osp
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
        description='Visualize RQFormer RRoI attention heatmaps')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('img', help='image file or image directory')
    parser.add_argument('--out-dir', default='work_dirs/attention_heatmaps')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.45)
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
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


def patch_rroi_attention(model):
    """Patch RRoIAttention modules to store the last attention weights."""
    patched = []

    for module in model.modules():
        if module.__class__.__name__ != 'RRoIAttention':
            continue

        original_forward = module.forward

        def forward_with_cache(query, roi_feat, _module=module):
            bs, num_queries = query.shape[:2]
            attn = _module.attention_weights(query).view(
                bs, num_queries, _module.num_heads,
                _module.roi_pooler_resolution**2)
            attn = attn.softmax(-1)
            _module.last_attention_weights = attn.detach().cpu()

            attn = attn.unsqueeze(-2)
            value = _module.value_proj(
                roi_feat.permute(0, 2, 3, 1)).permute(0, 3, 1,
                                                      2).contiguous()
            value = value.view(bs, num_queries, _module.num_heads, -1,
                               _module.roi_pooler_resolution**2)
            output = (value * attn).sum(-1).view(bs, num_queries,
                                                 _module.embed_dims)
            return _module.output_proj(output)

        module.forward = forward_with_cache
        module._original_forward = original_forward
        patched.append(module)

    if not patched:
        raise RuntimeError('No RRoIAttention module found in the model.')
    return patched


def get_last_attention_map(model):
    maps = []
    for module in model.modules():
        attn = getattr(module, 'last_attention_weights', None)
        if attn is not None:
            maps.append(attn)

    if not maps:
        return None

    # Use the last decoder layer attention, average batch/query/head.
    attn = maps[-1].float()
    grid_size = int(np.sqrt(attn.shape[-1]))
    heatmap = attn.mean(dim=(0, 1, 2)).reshape(grid_size, grid_size).numpy()
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-6)
    return heatmap


def draw_rotated_boxes(image_bgr, pred_instances, score_thr):
    if pred_instances is None or len(pred_instances) == 0:
        return image_bgr

    scores = pred_instances.scores.detach().cpu().numpy()
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
        cv2.polylines(image_bgr, [poly.astype(np.int32)], True, color, 2)
    return image_bgr


def overlay_heatmap(image_bgr, heatmap, alpha):
    h, w = image_bgr.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_u8 = np.uint8(np.clip(heatmap * 255, 0, 255))
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)


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
    patch_rroi_attention(model)

    img_paths = collect_images(args.img)
    if args.max_images > 0:
        img_paths = img_paths[:args.max_images]

    for img_path in img_paths:
        result = inference_detector(model, img_path)
        image = mmcv.imread(img_path)
        heatmap = get_last_attention_map(model)

        if heatmap is not None:
            image = overlay_heatmap(image, heatmap, args.alpha)
        image = draw_rotated_boxes(image, result.pred_instances,
                                   args.score_thr)

        out_name = osp.splitext(osp.basename(img_path))[0] + '_rroi_heatmap.jpg'
        out_path = osp.join(args.out_dir, out_name)
        mmcv.imwrite(image, out_path)
        print(f'[SAVED] {out_path}')


if __name__ == '__main__':
    main()
