#!/usr/bin/env python3
"""Post-training L2 PGD empirical-robustness evaluator.

`run_dir` may be a parent directory (containing settings.yml and timestamp subdirs)
or a single timestamp.<jobid> directory. dumped_setting.yml is preferred over
settings.yml since it has the resolved random-search values.
"""
import logging
import os
import sys
from argparse import ArgumentParser

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from parsers import SettingParser
from evaluator.pgd_attack import EvaluatePGD


# {36, 72, 108, 255}/255 — matches make_tree.py and the val_robstacc_* eval metrics.
DEFAULT_EPS = [0.1411, 0.2824, 0.4235, 1.0]


def parse_args():
    p = ArgumentParser('Post-training L2 PGD empirical-robustness evaluator.')
    p.add_argument('run_dir', help='Path to a trained run directory.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--eps', type=float, nargs='+', default=DEFAULT_EPS)
    p.add_argument('--n-iter', type=int, default=20)
    p.add_argument('--step-size', type=float, default=None,
                   help='Default Madry alpha = 2.5 * eps / n_iter.')
    p.add_argument('--no-random-start', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--output', default=None,
                   help='CSV path. Default: <run_dir>/pgd_metrics.csv.')
    return p.parse_args()


def resolve_run(run_dir: str) -> tuple:
    direct_state = os.path.join(run_dir, 'model_state_dict.pth')
    direct_dumped = os.path.join(run_dir, 'dumped_setting.yml')
    if os.path.isfile(direct_state) and os.path.isfile(direct_dumped):
        return direct_dumped, direct_state, run_dir

    candidates = []
    if os.path.isdir(run_dir):
        for sub in os.listdir(run_dir):
            sub_path = os.path.join(run_dir, sub)
            sd = os.path.join(sub_path, 'model_state_dict.pth')
            ds = os.path.join(sub_path, 'dumped_setting.yml')
            if os.path.isfile(sd) and os.path.isfile(ds):
                candidates.append((os.path.getmtime(sd), ds, sd, sub_path))
    if not candidates:
        raise FileNotFoundError(
            f'No model_state_dict.pth + dumped_setting.yml pair under {run_dir}.')
    candidates.sort(reverse=True)
    _, ds, sd, sub = candidates[0]
    return ds, sd, sub


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    log = logging.getLogger('pgd_eval')

    torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    settings_path, state_dict_path, output_dir = resolve_run(args.run_dir)
    log.info(f'Settings:    {settings_path}')
    log.info(f'State dict:  {state_dict_path}')

    parser = SettingParser(settings_path)
    parser.update_setting(f'device: {args.device}')
    parser.load_setting()
    cfg = parser.to_dict()

    if 'model' not in cfg:
        log.error('Settings file is missing model.')
        sys.exit(1)

    # Prefer explicit valset; otherwise the official test split of the trainset.
    if 'valset' in cfg:
        eval_set = cfg['valset']
        log.info(f'Using valset from settings: {type(eval_set).__name__}')
    elif 'trainset' in cfg:
        train_cls = type(cfg['trainset'])
        eval_set = train_cls(train=False, center=True)
        log.info(
            f'No valset; using {train_cls.__name__}(train=False) ({len(eval_set)} samples).'
        )
    else:
        log.error('Settings file has neither valset nor trainset.')
        sys.exit(1)

    device = torch.device(args.device)
    model = cfg['model'].to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()

    output_path = args.output or os.path.join(output_dir, 'pgd_metrics.csv')
    evaluator = EvaluatePGD(
        model=model, valset=eval_set, device=device,
        eps_list=args.eps, n_iter=args.n_iter, step_size=args.step_size,
        random_start=not args.no_random_start,
        batch_size=args.batch_size, num_workers=args.num_workers, logger=log,
    )
    summary = evaluator.run(output_path)
    log.info('=== Empirical RA summary ===')
    for k, v in summary.items():
        log.info(f'  {k:>20s} = {v:.4f}')


if __name__ == '__main__':
    main()
