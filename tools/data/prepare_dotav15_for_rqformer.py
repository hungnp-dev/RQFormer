"""Prepare raw DOTA-v1.5 data for the RQFormer configs.

The RQFormer DOTA-v1.5 config expects:

    data/split_ss_dota1_5/
      trainval/
        images/*.png
        annfiles/*.txt

Raw DOTA labels contain two metadata lines and large images. This script uses
the v1.5 oriented labels, removes metadata, splits images into 1024 patches,
and rewrites polygon coordinates for each patch.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image


DOTAV15_CLASSES = {
    'plane',
    'baseball-diamond',
    'bridge',
    'ground-track-field',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'tennis-court',
    'basketball-court',
    'storage-tank',
    'soccer-ball-field',
    'roundabout',
    'harbor',
    'swimming-pool',
    'helicopter',
    'container-crane',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare DOTA-v1.5 for RQFormer/MMRotate.')
    parser.add_argument(
        '--src',
        default=r'H:\Master\Bao\datasets\DOTA-v1.5',
        help='Raw DOTA-v1.5 root.')
    parser.add_argument(
        '--dst',
        default='data/split_ss_dota1_5',
        help='Output root expected by configs/_base_/datasets/dotav15.py.')
    parser.add_argument('--patch-size', type=int, default=1024)
    parser.add_argument('--gap', type=int, default=200)
    parser.add_argument(
        '--min-area-ratio',
        type=float,
        default=0.7,
        help='Keep clipped objects whose area ratio is at least this value.')
    parser.add_argument(
        '--include-empty',
        action='store_true',
        help='Also write image patches without objects.')
    return parser.parse_args()


def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def clip_polygon(poly: list[tuple[float, float]], box: tuple[int, int, int, int]):
    """Clip a polygon by an axis-aligned box using Sutherland-Hodgman."""
    xmin, ymin, xmax, ymax = box

    def clip_edge(points, inside, intersect):
        if not points:
            return []
        clipped = []
        prev = points[-1]
        prev_inside = inside(prev)
        for cur in points:
            cur_inside = inside(cur)
            if cur_inside:
                if not prev_inside:
                    clipped.append(intersect(prev, cur))
                clipped.append(cur)
            elif prev_inside:
                clipped.append(intersect(prev, cur))
            prev, prev_inside = cur, cur_inside
        return clipped

    def ix_with_x(x):
        return lambda p1, p2: (
            x,
            p1[1] + (p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0]),
        )

    def ix_with_y(y):
        return lambda p1, p2: (
            p1[0] + (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1]),
            y,
        )

    points = poly
    points = clip_edge(points, lambda p: p[0] >= xmin, ix_with_x(xmin))
    points = clip_edge(points, lambda p: p[0] <= xmax, ix_with_x(xmax))
    points = clip_edge(points, lambda p: p[1] >= ymin, ix_with_y(ymin))
    points = clip_edge(points, lambda p: p[1] <= ymax, ix_with_y(ymax))
    return points


def min_area_quad(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Approximate cv2.minAreaRect without adding an OpenCV dependency."""
    best = None
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        angle = math.atan2(y2 - y1, x2 - x1)
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        rotated = [
            (x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in poly
        ]
        xs = [p[0] for p in rotated]
        ys = [p[1] for p in rotated]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        area = (xmax - xmin) * (ymax - ymin)
        if best is None or area < best[0]:
            best = (area, angle, xmin, ymin, xmax, ymax)

    _, angle, xmin, ymin, xmax, ymax = best
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rect = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in rect]


def read_dota_label(path: Path):
    objects = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts or parts[0].startswith('imagesource') or parts[0].startswith('gsd'):
            continue
        if len(parts) < 10:
            raise ValueError(f'Bad annotation line {path}:{line_no}: {line}')
        cls_name = parts[8]
        if cls_name not in DOTAV15_CLASSES:
            raise ValueError(f'Unknown DOTA-v1.5 class {cls_name!r} in {path}')
        coords = [float(x) for x in parts[:8]]
        poly = list(zip(coords[0::2], coords[1::2]))
        objects.append((poly, cls_name, parts[9]))
    return objects


def iter_windows(width: int, height: int, size: int, gap: int):
    step = size - gap
    x_num = 1 if width <= size else math.ceil((width - size) / step) + 1
    y_num = 1 if height <= size else math.ceil((height - size) / step) + 1
    xs = [step * i for i in range(x_num)]
    ys = [step * i for i in range(y_num)]
    if xs and xs[-1] + size > width:
        xs[-1] = max(width - size, 0)
    if ys and ys[-1] + size > height:
        ys[-1] = max(height - size, 0)
    for x in sorted(set(xs)):
        for y in sorted(set(ys)):
            yield x, y, x + size, y + size


def collect_split(src: Path, split: str):
    image_root = src / split / 'images'
    label_root = src / split / 'labelTxt-v1.5' / f'DOTA-v1.5_{split}'
    images = {p.stem: p for p in image_root.rglob('*.png')}
    labels = {p.stem: p for p in label_root.glob('*.txt')}
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels:
        raise RuntimeError(
            f'{split}: {len(missing_labels)} images have no label, e.g. '
            f'{missing_labels[:5]}')
    if missing_images:
        raise RuntimeError(
            f'{split}: {len(missing_images)} labels have no image, e.g. '
            f'{missing_images[:5]}')
    return [(img_id, images[img_id], labels[img_id]) for img_id in sorted(images)]


def format_obj(poly, cls_name: str, difficulty: str, x0: int, y0: int) -> str:
    shifted = [(x - x0, y - y0) for x, y in poly]
    flat = []
    for x, y in shifted:
        flat.extend([f'{x:.1f}', f'{y:.1f}'])
    return ' '.join(flat + [cls_name, difficulty])


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    out_images = dst / 'trainval' / 'images'
    out_ann = dst / 'trainval' / 'annfiles'
    out_images.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    items = collect_split(src, 'train') + collect_split(src, 'val')
    total_patches = 0
    total_objects = 0

    for index, (img_id, image_path, label_path) in enumerate(items, start=1):
        objects = read_dota_label(label_path)
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            width, height = image.size
            for x0, y0, x1, y1 in iter_windows(
                    width, height, args.patch_size, args.gap):
                lines = []
                window = (x0, y0, min(x1, width), min(y1, height))
                for poly, cls_name, difficulty in objects:
                    original_area = polygon_area(poly)
                    clipped = clip_polygon(poly, window)
                    clipped_area = polygon_area(clipped)
                    if original_area <= 0 or clipped_area <= 0:
                        continue
                    if clipped_area / original_area < args.min_area_ratio:
                        continue
                    if len(clipped) != 4:
                        clipped = min_area_quad(clipped)
                    lines.append(format_obj(clipped, cls_name, difficulty, x0, y0))

                if not lines and not args.include_empty:
                    continue

                patch_id = f'{img_id}__1.0__{x0}___{y0}'
                patch = Image.new('RGB', (args.patch_size, args.patch_size))
                crop = image.crop((x0, y0, min(x1, width), min(y1, height)))
                patch.paste(crop, (0, 0))
                patch.save(out_images / f'{patch_id}.png')
                (out_ann / f'{patch_id}.txt').write_text('\n'.join(lines) + '\n')
                total_patches += 1
                total_objects += len(lines)

        if index % 100 == 0:
            print(f'Processed {index}/{len(items)} source images...')

    print(f'Done. Wrote {total_patches} patches and {total_objects} objects.')
    print(f'Images: {out_images}')
    print(f'Annotations: {out_ann}')


if __name__ == '__main__':
    main()
