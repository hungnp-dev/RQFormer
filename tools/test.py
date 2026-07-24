# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import glob
import json
import os
import os.path as osp
import logging

import mmcv
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmdet.registry import HOOKS
from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmrotate.utils import register_all_modules


DEFAULT_CONFIG = (
    'projects/RQFormer/configs/'
    'rroiformer_le90_r50_q500_layer2_sq1_dq1_t0.85_3x_dior.py')
DEFAULT_SHOW_DIR = 'test_bbox_visualizations'
DEFAULT_PREDICTION_FILE = 'test_predictions.pkl'
DEFAULT_METRICS_FILE = 'test_metrics.json'
DEFAULT_METRICS_CHART = 'test_metrics_chart.png'
DEFAULT_IMAGES_PER_CLASS = 2
DEFAULT_VIS_SCORE_THR = 0.3


@HOOKS.register_module()
class TopKPerClassVisualizationHook(Hook):
    """Export only the highest-confidence test images for each class."""

    def __init__(self,
                 test_out_dir=DEFAULT_SHOW_DIR,
                 images_per_class=DEFAULT_IMAGES_PER_CLASS,
                 score_thr=DEFAULT_VIS_SCORE_THR,
                 show=False,
                 wait_time=0.,
                 backend_args=None):
        self._visualizer = Visualizer.get_current_instance()
        self.test_out_dir = test_out_dir
        self.images_per_class = images_per_class
        self.score_thr = score_thr
        self.show = show
        self.wait_time = wait_time
        self.backend_args = backend_args
        self.best_by_class = {}

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        if self.test_out_dir is None or self.images_per_class <= 0:
            return

        for data_sample in outputs:
            pred_instances = data_sample.pred_instances
            if 'scores' not in pred_instances or len(pred_instances) == 0:
                continue

            scores = pred_instances.scores.detach().cpu()
            labels = pred_instances.labels.detach().cpu()
            for label in labels.unique().tolist():
                label_mask = labels == label
                label_scores = scores[label_mask]
                best_score = float(label_scores.max())
                if best_score < self.score_thr:
                    continue

                item = dict(
                    score=best_score,
                    label=int(label),
                    img_path=data_sample.img_path,
                    data_sample=copy.deepcopy(data_sample).cpu())
                class_items = self.best_by_class.setdefault(int(label), [])
                class_items.append(item)
                class_items.sort(key=lambda x: x['score'], reverse=True)
                del class_items[self.images_per_class:]

    def after_test_epoch(self, runner, metrics=None):
        if self.test_out_dir is None or not self.best_by_class:
            print_log(
                'No high-confidence bounding-box visualization was selected.',
                logger='current',
                level=logging.WARNING)
            return

        out_root = osp.join(runner.work_dir, runner.timestamp,
                            self.test_out_dir)
        mkdir_or_exist(out_root)
        classes = runner.visualizer.dataset_meta.get('classes', None)
        exported = 0

        for label, items in sorted(self.best_by_class.items()):
            class_name = (
                classes[label] if classes is not None and label < len(classes)
                else f'class_{label}')
            safe_class_name = str(class_name).replace('/', '_').replace('\\', '_')
            class_dir = osp.join(out_root, safe_class_name)
            mkdir_or_exist(class_dir)

            for rank, item in enumerate(items, start=1):
                img_bytes = get(
                    item['img_path'], backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                img_name = osp.splitext(osp.basename(item['img_path']))[0]
                out_file = osp.join(
                    class_dir,
                    f'rank{rank:02d}_score{item["score"]:.4f}_{img_name}.jpg')
                data_sample = copy.deepcopy(item['data_sample'])
                pred_instances = data_sample.pred_instances
                keep = (
                    (pred_instances.labels == item['label']) &
                    (pred_instances.scores >= self.score_thr))
                data_sample.pred_instances = pred_instances[keep]
                self._visualizer.add_datasample(
                    f'{safe_class_name}_rank{rank:02d}',
                    img,
                    data_sample=data_sample,
                    show=self.show,
                    wait_time=self.wait_time,
                    pred_score_thr=self.score_thr,
                    out_file=out_file,
                    step=exported)
                exported += 1

        print_log(
            f'Exported {exported} selected bounding-box images to {out_root}',
            logger='current')


def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument(
        'config',
        nargs='?',
        default=DEFAULT_CONFIG,
        help='test config file path. Defaults to the DIOR-R CPSQ config')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        help='checkpoint file. If omitted, the newest best_*.pth/.pt in '
        'the work_dir is used automatically')
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'auto'],
        default='cuda',
        help='device used for testing')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='visible GPU index used when --cuda-visible-devices is not set')
    parser.add_argument(
        '--cuda-visible-devices',
        help='value assigned to CUDA_VISIBLE_DEVICES, for example "0" or "0,1"')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation. '
        'Defaults to work_dir/test_predictions.pkl')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--metrics-json',
        help='path to save final test metrics as JSON. Defaults to '
        'work_dir/test_metrics.json')
    parser.add_argument(
        '--metrics-chart',
        help='path to save a bar chart of numeric metrics. Defaults to '
        'work_dir/test_metrics_chart.png')
    parser.add_argument(
        '--images-per-class',
        type=int,
        default=DEFAULT_IMAGES_PER_CLASS,
        help='number of highest-confidence visualization images exported '
        'for each class')
    parser.add_argument(
        '--vis-score-thr',
        type=float,
        default=DEFAULT_VIS_SCORE_THR,
        help='minimum prediction score used for selected bbox visualization')
    parser.add_argument(
        '--no-auto-output',
        action='store_true',
        help='disable automatic prediction dump, metric JSON, metric chart, '
        'and bbox visualization defaults')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    elif args.gpu_id is not None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def configure_device_for_windows_gpu(cfg, args):
    """Configure single-GPU CUDA testing cleanly on Windows."""
    if os.name == 'nt':
        cfg.setdefault('env_cfg', {})
        cfg.env_cfg.setdefault('mp_cfg', {})
        cfg.env_cfg.mp_cfg['mp_start_method'] = 'spawn'
        cfg.env_cfg.mp_cfg.setdefault('opencv_num_threads', 0)
        cfg.env_cfg.setdefault('dist_cfg', {})
        cfg.env_cfg.dist_cfg['backend'] = 'gloo'

    import torch

    cuda_available = torch.cuda.is_available()
    if args.device == 'cuda' and not cuda_available:
        raise RuntimeError(
            'CUDA is not available. Check your NVIDIA driver, CUDA-enabled '
            'PyTorch install, and CUDA_VISIBLE_DEVICES. If you intentionally '
            'want CPU testing, pass --device cpu.')

    device = 'cuda' if args.device == 'auto' and cuda_available else args.device
    if device == 'cuda':
        torch.cuda.set_device(args.local_rank)
        cfg.device = 'cuda'
        print_log(
            'GPU testing enabled: '
            f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}, '
            f'current_device={torch.cuda.current_device()}, '
            f'name={torch.cuda.get_device_name(torch.cuda.current_device())}',
            logger='current')
    else:
        cfg.device = 'cpu'
        print_log(
            'CPU testing enabled. This is usually very slow for RQFormer.',
            logger='current',
            level=logging.WARNING)


def configure_dataloader_for_windows(cfg):
    """Make dataloader settings valid for Windows and low-worker runs."""
    for loader_name in ['val_dataloader', 'test_dataloader']:
        dataloader = cfg.get(loader_name, None)
        if dataloader is None:
            continue
        if dataloader.get('num_workers', None) == 0:
            dataloader['persistent_workers'] = False


def configure_terminal_logging(cfg):
    """Print every test iteration/progress message to the terminal."""
    cfg.log_level = 'INFO'
    cfg.setdefault('default_hooks', {})
    cfg.default_hooks.setdefault('logger', dict(type='LoggerHook'))
    cfg.default_hooks.logger['interval'] = 1
    cfg.default_hooks.logger.setdefault('ignore_last', False)


def find_best_checkpoint(work_dir):
    """Find the best checkpoint saved by MMEngine in a work directory."""
    patterns = [
        osp.join(work_dir, 'best_*.pth'),
        osp.join(work_dir, 'best_*.pt'),
        osp.join(work_dir, '**', 'best_*.pth'),
        osp.join(work_dir, '**', 'best_*.pt'),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))
    if not candidates:
        fallback_patterns = [
            osp.join(work_dir, 'latest.pth'),
            osp.join(work_dir, 'latest.pt'),
            osp.join(work_dir, 'epoch_*.pth'),
            osp.join(work_dir, 'epoch_*.pt'),
        ]
        for pattern in fallback_patterns:
            candidates.extend(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            'No checkpoint was provided and no best/latest/epoch checkpoint '
            f'was found in {work_dir}. Train first or pass a checkpoint path.')
    return max(candidates, key=osp.getmtime)


def configure_auto_outputs(cfg, args):
    """Enable full test artifacts by default."""
    if args.no_auto_output:
        return
    if args.out is None:
        args.out = osp.join(cfg.work_dir, DEFAULT_PREDICTION_FILE)
    if args.metrics_json is None:
        args.metrics_json = osp.join(cfg.work_dir, DEFAULT_METRICS_FILE)
    if args.metrics_chart is None:
        args.metrics_chart = osp.join(cfg.work_dir, DEFAULT_METRICS_CHART)
    if args.show_dir is None:
        args.show_dir = DEFAULT_SHOW_DIR


def disable_backbone_pretrain_for_testing(cfg):
    """Avoid downloading torchvision weights when a test checkpoint is loaded."""
    backbone = cfg.get('model', {}).get('backbone', None)
    if isinstance(backbone, dict) and backbone.get('init_cfg', None) is not None:
        backbone['init_cfg'] = None


def summarize_test_dataset(cfg):
    dataset = cfg.get('test_dataloader', {}).get('dataset', {})
    while isinstance(dataset, dict) and 'dataset' in dataset:
        dataset = dataset['dataset']
    if not isinstance(dataset, dict):
        return 'unknown'

    parts = [f"type={dataset.get('type', 'unknown')}"]
    for key in ['data_root', 'ann_file']:
        if key in dataset:
            parts.append(f'{key}={dataset[key]}')
    if 'data_prefix' in dataset:
        parts.append(f"data_prefix={dataset['data_prefix']}")
    return ', '.join(parts)


def log_test_summary(cfg, args):
    """Print a compact startup summary before testing starts."""
    print('\nCPSQ-RQFormer test startup', flush=True)
    print(f'Config: {args.config}', flush=True)
    print(f'Checkpoint: {args.checkpoint}', flush=True)
    print(f'Work dir: {cfg.work_dir}', flush=True)
    print(f'Device: {cfg.get("device", "unknown")}', flush=True)
    print(f'Launcher: {cfg.launcher}', flush=True)
    print(f'Test dataset: {summarize_test_dataset(cfg)}', flush=True)
    print(f'Evaluator: {cfg.get("test_evaluator", "unknown")}', flush=True)
    print(f'Prediction dump: {args.out}', flush=True)
    print(f'Visualization dir: {args.show_dir}', flush=True)
    print(f'Visualization images/class: {args.images_per_class}', flush=True)
    print(f'Visualization score threshold: {args.vis_score_thr}', flush=True)
    print(f'Metrics JSON: {args.metrics_json}', flush=True)
    print(f'Metrics chart: {args.metrics_chart}', flush=True)
    print('Terminal test log: enabled\n', flush=True)

    print_log('=' * 80, logger='current')
    print_log('CPSQ-RQFormer test startup', logger='current')
    print_log(f'Config: {args.config}', logger='current')
    print_log(f'Checkpoint: {args.checkpoint}', logger='current')
    print_log(f'Work dir: {cfg.work_dir}', logger='current')
    print_log(f'Device: {cfg.get("device", "unknown")}', logger='current')
    print_log(f'Launcher: {cfg.launcher}', logger='current')
    print_log(f'Test dataset: {summarize_test_dataset(cfg)}',
              logger='current')
    print_log(f'Evaluator: {cfg.get("test_evaluator", "unknown")}',
              logger='current')
    print_log(f'Prediction dump: {args.out}', logger='current')
    print_log(f'Visualization dir: {args.show_dir}', logger='current')
    print_log(f'Visualization images/class: {args.images_per_class}',
              logger='current')
    print_log(f'Visualization score threshold: {args.vis_score_thr}',
              logger='current')
    print_log(f'Metrics JSON: {args.metrics_json}', logger='current')
    print_log(f'Metrics chart: {args.metrics_chart}', logger='current')
    print_log('=' * 80, logger='current')


def json_safe(value):
    """Convert metric values to JSON-serializable Python objects."""
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save_metrics_json(metrics, path):
    if path is None:
        return
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_safe(metrics), f, indent=2, ensure_ascii=False)
    print_log(f'Metrics JSON saved to {path}', logger='current')


def save_metrics_chart(metrics, path):
    if path is None:
        return
    numeric_metrics = {
        key: float(value)
        for key, value in json_safe(metrics).items()
        if isinstance(value, (int, float))
    }
    if not numeric_metrics:
        print_log(
            'No numeric metric was found, metrics chart was skipped.',
            logger='current',
            level=logging.WARNING)
        return

    os.makedirs(osp.dirname(path), exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(numeric_metrics.keys())
    values = list(numeric_metrics.values())
    fig_width = max(8, min(18, len(names) * 0.9))
    plt.figure(figsize=(fig_width, 5))
    plt.bar(names, values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('value')
    plt.title('Test metrics')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print_log(f'Metrics chart saved to {path}', logger='current')


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        default_hooks['visualization'] = dict(
            type='mmdet.TopKPerClassVisualizationHook',
            test_out_dir=args.show_dir,
            images_per_class=args.images_per_class,
            score_thr=args.vis_score_thr,
            show=args.show,
            wait_time=args.wait_time)
    else:
        default_hooks['visualization'] = dict(
            type='mmdet.TopKPerClassVisualizationHook',
            test_out_dir=args.show_dir,
            images_per_class=args.images_per_class,
            score_thr=args.vis_score_thr,
            show=args.show,
            wait_time=args.wait_time)

    return cfg


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    configure_device_for_windows_gpu(cfg, args)
    configure_terminal_logging(cfg)
    configure_dataloader_for_windows(cfg)
    disable_backbone_pretrain_for_testing(cfg)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.checkpoint is None:
        args.checkpoint = find_best_checkpoint(cfg.work_dir)
    configure_auto_outputs(cfg, args)
    cfg.load_from = args.checkpoint
    log_test_summary(cfg, args)

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # start testing
    print_log('Test loop is starting now.', logger='current')
    metrics = runner.test()
    print_log(f'Full test metrics: {metrics}', logger='current')
    print(f'\nFull test metrics:\n{metrics}\n', flush=True)
    save_metrics_json(metrics, args.metrics_json)
    save_metrics_chart(metrics, args.metrics_chart)


if __name__ == '__main__':
    main()
