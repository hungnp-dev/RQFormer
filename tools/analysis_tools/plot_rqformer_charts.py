import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot compact RQFormer charts from MMEngine JSON logs.')
    parser.add_argument(
        'json_logs', nargs='+', help='MMEngine JSON log files.')
    parser.add_argument(
        '--out-dir', required=True, help='Directory to save chart images.')
    parser.add_argument('--title', default='RQFormer', help='Chart title.')
    return parser.parse_args()


def load_records(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def collect_series(records):
    series = defaultdict(list)
    for idx, record in enumerate(records):
        step = record.get('step', record.get('iter', idx))
        epoch = record.get('epoch')
        x = step if step is not None else idx
        if epoch is not None and step is not None:
            x = float(epoch) + float(step) / 1000000.0
        for key, value in record.items():
            number = as_number(value)
            if number is None:
                continue
            if key in {'step', 'iter', 'epoch'}:
                continue
            series[key].append((x, number))
    return series


def normalize_key(key):
    return key.split('/')[-1]


def select_keys(series, patterns):
    keys = []
    for key in sorted(series):
        key_l = key.lower()
        base_l = normalize_key(key).lower()
        if any(pattern in key_l or pattern in base_l for pattern in patterns):
            keys.append(key)
    return keys


def plot_group(series, keys, out_file, title, ylabel):
    keys = [key for key in keys if len(series[key]) >= 2]
    if not keys:
        return False

    plt.figure(figsize=(11, 6))
    for key in keys:
        points = series[key]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, linewidth=1.2, label=normalize_key(key))

    plt.title(title)
    plt.xlabel('epoch/step')
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.4, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    return True


def write_summary(series, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('metric,count,last,min,max\n')
        for key in sorted(series):
            values = [v for _, v in series[key]]
            if not values:
                continue
            f.write(f'{key},{len(values)},{values[-1]},{min(values)},{max(values)}\n')


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_series = defaultdict(list)
    for log in args.json_logs:
        path = Path(log)
        if not path.is_file():
            continue
        records = load_records(path)
        series = collect_series(records)
        stem = path.stem
        for key, values in series.items():
            scoped_key = f'{stem}/{key}'
            all_series[scoped_key].extend(values)

    if not all_series:
        raise SystemExit('No numeric metrics found in JSON logs.')

    write_summary(all_series, out_dir / 'metrics_summary.csv')

    generated = []
    groups = [
        ('loss_curves.png', ['loss'], 'Loss curves', 'loss'),
        ('learning_rate.png', ['lr', 'learning_rate'], 'Learning rate', 'lr'),
        ('eval_metrics.png', ['map', 'ap50', 'ap75', 'recall', 'precision'],
         'Evaluation metrics', 'metric'),
        ('runtime_stats.png', ['time', 'data_time', 'memory'],
         'Runtime statistics', 'value'),
    ]

    for filename, patterns, group_title, ylabel in groups:
        keys = select_keys(all_series, patterns)
        out_file = out_dir / filename
        if plot_group(
                all_series, keys, out_file,
                f'{args.title} - {group_title}', ylabel):
            generated.append(str(out_file))

    with open(out_dir / 'generated_charts.txt', 'w', encoding='utf-8') as f:
        for path in generated:
            f.write(path + '\n')

    print('Generated charts:')
    for path in generated:
        print(path)


if __name__ == '__main__':
    main()
