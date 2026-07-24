# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules


TERMINAL_LOG_INTERVAL = 1


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'auto'],
        default='cuda',
        help='device used for fine-tuning. Use cuda for Windows GPU training')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='visible GPU index used when --cuda-visible-devices is not set')
    parser.add_argument(
        '--cuda-visible-devices',
        help='value assigned to CUDA_VISIBLE_DEVICES, for example "0" or "0,1"')
    parser.add_argument(
        '--load-from',
        help='checkpoint path used as the fine-tuning starting point')
    parser.add_argument(
        '--freeze-original',
        action='store_true',
        default=True,
        help='freeze all existing model parameters before fine-tuning')
    parser.add_argument(
        '--no-freeze-original',
        action='store_false',
        dest='freeze_original',
        help='disable full original-model freezing')
    parser.add_argument(
        '--trainable-keywords',
        nargs='+',
        default=['cpsq'],
        help='parameter-name keywords that should stay trainable after '
        'freezing the original model')
    parser.add_argument(
        '--allow-empty-trainable',
        action='store_true',
        help='allow training to start even when no parameter matches '
        '--trainable-keywords')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings with key=value pairs')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    elif args.gpu_id is not None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def configure_device_for_windows_gpu(cfg, args):
    """Configure single-GPU CUDA training cleanly on Windows."""
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
            'want CPU training, pass --device cpu.')

    device = 'cuda' if args.device == 'auto' and cuda_available else args.device
    if device == 'cuda':
        torch.cuda.set_device(args.local_rank)
        cfg.device = 'cuda'
        print_log(
            'GPU fine-tuning enabled: '
            f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}, '
            f'current_device={torch.cuda.current_device()}, '
            f'name={torch.cuda.get_device_name(torch.cuda.current_device())}',
            logger='current')
    else:
        cfg.device = 'cpu'
        print_log(
            'CPU fine-tuning enabled. This is usually very slow for RQFormer.',
            logger='current',
            level=logging.WARNING)


def configure_terminal_logging(cfg):
    """Ensure fine-tuning progress is printed clearly in the terminal."""
    cfg.log_level = 'INFO'
    cfg.setdefault('default_hooks', {})
    cfg.default_hooks.setdefault('logger', dict(type='LoggerHook'))
    cfg.default_hooks.logger['interval'] = TERMINAL_LOG_INTERVAL
    cfg.default_hooks.logger.setdefault('ignore_last', False)

    cfg.setdefault('log_processor', dict(type='LogProcessor'))
    cfg.log_processor['window_size'] = TERMINAL_LOG_INTERVAL
    cfg.log_processor.setdefault('by_epoch', True)


def configure_dataloader_for_windows(cfg):
    """Make dataloader settings valid for Windows and low-worker runs."""
    for loader_name in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
        dataloader = cfg.get(loader_name, None)
        if dataloader is None:
            continue
        if dataloader.get('num_workers', None) == 0:
            dataloader['persistent_workers'] = False


def configure_best_checkpoint_saving(cfg):
    """Save the best validation checkpoint during fine-tuning by default."""
    cfg.setdefault('default_hooks', {})
    cfg.default_hooks.setdefault('checkpoint', dict(type='CheckpointHook'))
    cfg.default_hooks.checkpoint.setdefault('interval', 1)
    cfg.default_hooks.checkpoint['save_best'] = 'auto'
    cfg.default_hooks.checkpoint['rule'] = 'greater'
    cfg.default_hooks.checkpoint.setdefault('max_keep_ckpts', 3)


def disable_backbone_pretrain_for_finetuning(cfg):
    """Avoid downloading torchvision weights when a full checkpoint is loaded."""
    if cfg.get('load_from', None) is None:
        return
    backbone = cfg.get('model', {}).get('backbone', None)
    if isinstance(backbone, dict) and backbone.get('init_cfg', None) is not None:
        backbone['init_cfg'] = None


def summarize_train_dataset(cfg):
    dataset = cfg.get('train_dataloader', {}).get('dataset', {})
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


def log_finetuning_summary(cfg, args):
    """Print a compact startup summary before Runner construction."""
    print('\nCPSQ-RQFormer fine-tuning startup', flush=True)
    print(f'Config: {args.config}', flush=True)
    print(f'Work dir: {cfg.work_dir}', flush=True)
    print(f'Load from: {cfg.get("load_from", None)}', flush=True)
    print(f'Device: {cfg.get("device", "unknown")}', flush=True)
    print(f'Train dataset: {summarize_train_dataset(cfg)}', flush=True)
    print(
        f'Freeze original: {args.freeze_original}, '
        f'trainable keywords: {args.trainable_keywords}',
        flush=True)
    print('Terminal training log: enabled\n', flush=True)

    print_log('=' * 80, logger='current')
    print_log('CPSQ-RQFormer fine-tuning startup', logger='current')
    print_log(f'Config: {args.config}', logger='current')
    print_log(f'Work dir: {cfg.work_dir}', logger='current')
    print_log(f'Load from: {cfg.get("load_from", None)}', logger='current')
    print_log(f'Resume: {cfg.get("resume", False)}', logger='current')
    print_log(f'Device: {cfg.get("device", "unknown")}', logger='current')
    print_log(f'Launcher: {cfg.launcher}', logger='current')
    print_log(f'Log interval: every {TERMINAL_LOG_INTERVAL} iteration',
              logger='current')
    print_log(f'Train dataset: {summarize_train_dataset(cfg)}',
              logger='current')
    print_log(
        f'Freeze original: {args.freeze_original}, '
        f'trainable keywords: {args.trainable_keywords}',
        logger='current')
    print_log('=' * 80, logger='current')


def freeze_original_parameters(model, trainable_keywords, allow_empty=False):
    """Freeze original parameters and keep newly added modules trainable."""
    keywords = tuple(k.lower() for k in trainable_keywords)
    trainable = []
    frozen = []
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        is_trainable = any(keyword in name.lower() for keyword in keywords)
        param.requires_grad = is_trainable
        if is_trainable:
            trainable.append(name)
            trainable_params += param.numel()
        else:
            frozen.append(name)
            frozen_params += param.numel()

    print_log(
        f'Freeze original model: frozen={len(frozen)}, '
        f'trainable={len(trainable)}, keywords={list(trainable_keywords)}',
        logger='current')
    print_log(
        f'Parameter count: frozen={frozen_params:,}, '
        f'trainable={trainable_params:,}',
        logger='current')

    if trainable:
        preview = ', '.join(trainable[:20])
        if len(trainable) > 20:
            preview += ', ...'
        print_log(f'Trainable parameters: {preview}', logger='current')
    elif not allow_empty:
        raise RuntimeError(
            'No trainable parameters were found after freezing the original '
            'model. Check that the config contains roi_head.cpsq_cfg, pass '
            'the right --trainable-keywords, or use --no-freeze-original.')


def main():
    args = parse_args()

    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    configure_device_for_windows_gpu(cfg, args)
    configure_terminal_logging(cfg)
    configure_dataloader_for_windows(cfg)
    configure_best_checkpoint_saving(cfg)
    disable_backbone_pretrain_for_finetuning(cfg)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        print_log(
            'AMP is disabled for this RQFormer fine-tuning run because the '
            'original QualityFocalLoss uses binary_cross_entropy on '
            'probabilities, which is unsafe under autocast.',
            logger='current',
            level=logging.WARNING)
        args.amp = False

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    cfg.resume = args.resume
    log_finetuning_summary(cfg, args)

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    if args.freeze_original:
        freeze_original_parameters(
            runner.model,
            args.trainable_keywords,
            allow_empty=args.allow_empty_trainable)

    print_log('Fine-tuning loop is starting now.', logger='current')
    runner.train()
    print_log('Fine-tuning finished.', logger='current')


if __name__ == '__main__':
    main()
