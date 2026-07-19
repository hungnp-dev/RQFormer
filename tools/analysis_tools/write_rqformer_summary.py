#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path

JOBS = [
    dict(name='dior', dataset='DIOR-R', group='rroiformer', lr_schd='3x',
         config='projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.85_3x_dior.py'),
    dict(name='dotav1_0', dataset='DOTA-v1.0', group='rroiformer', lr_schd='2x',
         config='projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.0.py'),
    dict(name='dotav1_5', dataset='DOTA-v1.5', group='rroiformer', lr_schd='2x',
         config='projects/RQFormer/configs/rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.9_2x_dotav1.5.py'),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Write RQFormer summary table.')
    parser.add_argument('--repo-dir', default='.', help='RQFormer repo directory')
    parser.add_argument('--work-root', default='work_dirs', help='work_dirs path')
    parser.add_argument('--out', default=None, help='output markdown path')
    return parser.parse_args()


def read_text(path):
    return Path(path).read_text(encoding='utf-8', errors='ignore')


def find_config_value(text, pattern, default='-'):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def config_meta(repo, rel_config):
    cfg_path = repo / rel_config
    text = read_text(cfg_path) if cfg_path.is_file() else ''
    depth = find_config_value(text, r'depth\s*=\s*(\d+)', '-')
    query = find_config_value(text, r'num_proposals\s*=\s*(\d+)', '-')
    angle = find_config_value(text, r"angle_version\s*=\s*['\"]([^'\"]+)", '-')
    backbone = f'R{depth}' if depth != '-' else '-'
    return dict(backbone=backbone, query=query, angle=angle, config=rel_config)


def batch_from_log_or_config(repo, work_dir, rel_config):
    # Prefer actual runtime config dumped in work dir, then source/base config.
    candidates = list(work_dir.glob('train/**/*.py')) + list(work_dir.glob('test/**/*.py'))
    for path in candidates:
        text = read_text(path)
        m = re.search(r'train_dataloader\s*=\s*dict\([^\n]*batch_size\s*=\s*(\d+)', text)
        if m:
            return m.group(1)
        m = re.search(r'batch_size\s*=\s*(\d+)', text)
        if m and 'train_dataloader' in text:
            return m.group(1)

    # Source base dataset files in this cleaned repo all define train batch_size.
    cfg_text = read_text(repo / rel_config)
    if 'dior.py' in cfg_text:
        base = repo / 'configs/_base_/datasets/dior.py'
    elif 'dotav15.py' in cfg_text:
        base = repo / 'configs/_base_/datasets/dotav15.py'
    else:
        base = repo / 'configs/_base_/datasets/dota.py'
    text = read_text(base) if base.is_file() else ''
    m = re.search(r'train_dataloader\s*=\s*dict\(\s*batch_size\s*=\s*(\d+)', text, re.S)
    return m.group(1) if m else '-'


def normalize_metric_value(value):
    if value is None:
        return '-'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return '-'
    if value <= 1.5:
        value *= 100.0
    return f'{value:.2f}'


def collect_metrics_from_json(json_file):
    metrics = {}
    with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                key_l = key.lower().split('/')[-1]
                if key_l in {'ap50', 'ap75', 'map'} and isinstance(value, (int, float)):
                    metrics[key_l.upper() if key_l.startswith('ap') else 'mAP'] = value
    return metrics


def collect_metrics_from_text(log_file):
    metrics = {}
    text = read_text(log_file)
    patterns = {
        'AP50': r'(?:dota/)?AP50\s*[:=]\s*([0-9.]+)',
        'AP75': r'(?:dota/)?AP75\s*[:=]\s*([0-9.]+)',
        'mAP': r'(?:dota/)?mAP\s*[:=]\s*([0-9.]+)',
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def metrics_for(work_dir):
    metrics = {}
    for path in sorted(work_dir.glob('test/**/*.json')) + sorted(work_dir.glob('train/**/*.json')):
        metrics.update(collect_metrics_from_json(path))
    for path in sorted(work_dir.glob('test/**/*.log')) + sorted(work_dir.glob('logs/*.txt')):
        metrics.update(collect_metrics_from_text(path))
    return metrics


def main():
    args = parse_args()
    repo = Path(args.repo_dir).resolve()
    work_root = Path(args.work_root).resolve()
    out = Path(args.out).resolve() if args.out else work_root / 'rroiformer' / 'summary_results.md'
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for job in JOBS:
        work_dir = work_root / job['group'] / job['name']
        meta = config_meta(repo, job['config'])
        metric = metrics_for(work_dir)
        rows.append([
            job['dataset'],
            normalize_metric_value(metric.get('AP50')),
            normalize_metric_value(metric.get('AP75')),
            normalize_metric_value(metric.get('mAP')),
            meta['backbone'],
            job['lr_schd'],
            batch_from_log_or_config(repo, work_dir, job['config']),
            meta['angle'],
            meta['query'],
            meta['config'],
        ])

    headers = ['Dataset', 'AP50', 'AP75', 'mAP', 'Backbone', 'lr schd', 'batch', 'Angle', 'Query', 'Configs']
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[SUMMARY] wrote {out}')


if __name__ == '__main__':
    main()


