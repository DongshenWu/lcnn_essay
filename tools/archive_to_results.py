#!/usr/bin/env python3
"""Canonical archival of a finished run -> results/<DATASET>/<size>_<layer>_<act>_<loss>.{pth,txt}."""
from argparse import ArgumentParser
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from parsers import SettingParser
from evaluator.pgd_attack import EvaluatePGD


RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
DEFAULT_EPS = [0.1411, 0.2824, 0.4235, 1.0]

SIZE_ABBREV = {
    'ConvNetXS': 'XS', 'ConvNetS': 'S', 'ConvNetM': 'M', 'ConvNetL': 'L',
    'TConvNetXS': 'TXS', 'TConvNetS': 'TS', 'TConvNetM': 'TM', 'TConvNetL': 'TL',
}
LAYER_ABBREV = {
    'AOL': 'AOL', 'BCOP': 'BCOP', 'Cayley': 'Cayley',
    'CPL': 'CPL', 'LOT': 'LOT', 'SLL': 'SLL', 'SOC': 'SOC',
    'SpectralNorm': 'SN', 'Sandwich': 'Sand',
}
ACT_ABBREV = {
    'MaxMin': 'GS',
    'AbsoluteValue': 'AV',
    'Abs': 'AV',
    'LinearSpline': 'LS',
    'Householder': 'HH',
    'NActivation': 'NA',
}
LOSS_ABBREV = {
    'LipCrossEntropyLoss': 'CE',
    'CrossEntropyLoss': 'CE',
    'MulticlassHingeWithMargin': 'HM',
    'MultiMarginLoss': 'HM',
}


def parse_args():
    p = ArgumentParser('Archive a trained run into results/<DATASET>/<name>.{pth,txt}.')
    p.add_argument('run_dir')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eps', type=float, nargs='+', default=DEFAULT_EPS)
    p.add_argument('--n-iter', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=100,
                   help='Eval batch size for PGD and inference-memory measurement.')
    p.add_argument('--force', action='store_true',
                   help='Overwrite an existing archive with the same name.')
    return p.parse_args()


def resolve_run(run_dir: str):
    """Return (settings_yml, state_dict, source_dir); descends into the most-recent timestamp subdir if needed."""
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
            f'No model_state_dict.pth + dumped_setting.yml under {run_dir}.')
    candidates.sort(reverse=True)
    _, ds, sd, sub = candidates[0]
    return ds, sd, sub


def strip_layer_suffixes(name: str) -> str:
    for suffix in ('Dirac', 'Orth', '2t'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    for suffix in ('Conv2d', 'Conv'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def derive_name(cfg) -> tuple:
    model = cfg['model']
    size_full = type(model).__name__
    size = SIZE_ABBREV.get(size_full, size_full)

    conv_cls = model.hp.get_conv
    layer_full = conv_cls.__name__
    layer_key = strip_layer_suffixes(layer_full)
    layer = LAYER_ABBREV.get(layer_key, layer_key)

    act_cls = model.hp.get_activation
    # mapping-form !activation returns functools.partial; fall back to .func.__name__.
    act_full = getattr(act_cls, '__name__', None) or act_cls.func.__name__
    act = ACT_ABBREV.get(act_full, act_full)

    loss_full = type(cfg['loss']).__name__
    loss = LOSS_ABBREV.get(loss_full, loss_full)

    dataset = type(cfg['trainset']).__name__

    full_name = f'{size}_{layer}_{act}_{loss}'
    return (size_full, layer_full, act_full, loss_full,
            size, layer, act, loss,
            dataset, full_name)


def read_training_stats(run_dir: str, log: logging.Logger):
    """Pull wall-clock + tail-mean throughputs + final accuracies from training_statistics.csv."""
    csv_path = os.path.join(run_dir, 'training_statistics.csv')
    if not os.path.isfile(csv_path):
        log.warning(f'No training_statistics.csv at {csv_path}')
        return {}
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    out = {}
    out['wall_clock_seconds'] = float(
        (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds())
    n_tail = min(10, len(df))
    if 'train_throughput' in df.columns:
        out['train_throughput'] = float(df['train_throughput'].tail(n_tail).mean())
    if 'val_throughput' in df.columns:
        out['val_throughput'] = float(df['val_throughput'].tail(n_tail).mean())
    if 'val_accuracy' in df.columns:
        out['final_val_accuracy'] = float(df['val_accuracy'].iloc[-1])
    for col in df.columns:
        if col.startswith('val_robstacc_'):
            out[col] = float(df[col].iloc[-1])
    return out


def measure_memory(model, dataset, batch_size: int, loss_fn,
                   device: torch.device, mode: str, log: logging.Logger):
    """Peak GPU memory in MiB for one batch (None on CPU)."""
    if device.type != 'cuda':
        return None
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if mode == 'train':
        model.train()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    elif mode == 'eval':
        model.eval()
        with torch.no_grad():
            _ = model(x)
    else:
        raise ValueError(mode)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return peak / (1024 ** 2)


def fmt_seconds(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - 3600 * h - 60 * m
    return f'{h:d}h{m:02d}m{sec:05.2f}s'


def write_summary(out_path: str, *, full_name, dataset, source_run_dir,
                  source_state_dict, archive_ts,
                  size_full, layer_full, act_full, loss_full,
                  cfg, training_stats, memory_train, memory_eval,
                  pgd_summary, eps_list, args):
    lines = []
    lines.append(f'Model: {full_name}')
    lines.append(f'Dataset: {dataset}')
    lines.append(f'Source run: {source_run_dir}')
    lines.append(f'Source state_dict: {source_state_dict}')
    lines.append(f'Archived: {archive_ts}')
    lines.append('')
    lines.append('Architecture')
    lines.append('------------')
    lines.append(f'  size       = {size_full}')
    conv_first_name = type(cfg['model'].hp.get_conv_first).__name__ \
        if not callable(cfg['model'].hp.get_conv_first) \
        else cfg['model'].hp.get_conv_first.__name__
    conv_head_name = type(cfg['model'].hp.get_conv_head).__name__ \
        if not callable(cfg['model'].hp.get_conv_head) \
        else cfg['model'].hp.get_conv_head.__name__
    lines.append(f'  layer      = {layer_full}  '
                 f'(first: {conv_first_name}, head: {conv_head_name})')
    lines.append(f'  activation = {act_full}')
    lines.append(f'  loss       = {loss_full}')
    lines.append('')
    lines.append('Training')
    lines.append('--------')
    lines.append(f'  epochs        = {cfg.get("epochs", "?")}')
    lines.append(f'  batch_size    = {cfg.get("batch_size", "?")}')
    if 'wall_clock_seconds' in training_stats:
        lines.append(f'  wall_clock    = '
                     f'{fmt_seconds(training_stats["wall_clock_seconds"])} '
                     f'({training_stats["wall_clock_seconds"]:.1f} s, last - first epoch)')
    lines.append('')
    lines.append('Throughput (samples/sec, mean of last 10 epochs)')
    lines.append('------------------------------------------------')
    if 'train_throughput' in training_stats:
        lines.append(f'  train = {training_stats["train_throughput"]:9.1f}')
    if 'val_throughput' in training_stats:
        lines.append(f'  eval  = {training_stats["val_throughput"]:9.1f}')
    lines.append('')
    lines.append('Peak GPU memory (MiB, single-batch on loaded model)')
    lines.append('---------------------------------------------------')
    bs = cfg.get('batch_size', '?')
    if memory_train is not None:
        lines.append(f'  train (forward+backward, batch={bs}) = {memory_train:9.1f}')
    else:
        lines.append('  train = N/A (CPU run)')
    if memory_eval is not None:
        lines.append(f'  eval  (forward only,    batch={args.batch_size}) = {memory_eval:9.1f}')
    else:
        lines.append('  eval  = N/A (CPU run)')
    lines.append('  note: excludes optimizer state buffers and DataLoader prefetch;')
    lines.append('        underestimates true train-loop peak by the optimizer-state term.')
    lines.append('')
    lines.append('Test-set robustness (PGD: n_iter=%d, alpha=2.5*eps/n_iter, random_start, seed=%d)' % (args.n_iter, args.seed))
    lines.append('-' * 80)
    lines.append(f'  clean accuracy = {pgd_summary["clean_acc"]*100:6.2f}%')
    lines.append('')
    lines.append(f'  {"eps":>10s}  {"radius (input units)":>22s}  {"CRA":>8s}  {"emp-RA":>8s}  {"gap":>7s}')
    any_negative_gap = False
    for eps in eps_list:
        cra = pgd_summary[f'cra_{eps:.4f}']
        emp = pgd_summary[f'emp_ra_{eps:.4f}']
        gap = emp - cra
        if gap < 0:
            any_negative_gap = True
        # Try to express eps as k/255 if it matches.
        k = round(eps * 255)
        radius = f'{k}/255' if abs(eps - k / 255) < 1e-3 else f'{eps:.4f}'
        lines.append(f'  {eps:>10.4f}  {radius:>22s}  '
                     f'{cra*100:7.2f}%  {emp*100:7.2f}%  {gap*100:+6.2f}pp')
    if any_negative_gap:
        lines.append('')
        lines.append('  WARNING: emp-RA < CRA at one or more eps. For a 1-Lipschitz layer this')
        lines.append('  must NEVER happen (PGD cannot beat a valid certificate). A negative gap')
        lines.append('  here indicates the layer is NOT actually 1-Lipschitz, so the CRA values')
        lines.append('  above are not certified -- treat them as informational only and rely on')
        lines.append('  emp-RA for this models true adversarial robustness.')
    lines.append('')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S')
    log = logging.getLogger('archive')

    torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    settings_path, state_dict_path, source_dir = resolve_run(args.run_dir)
    log.info(f'Settings:    {settings_path}')
    log.info(f'State dict:  {state_dict_path}')

    parser = SettingParser(settings_path)
    parser.update_setting(f'device: {args.device}')
    parser.load_setting()
    cfg = parser.to_dict()

    (size_full, layer_full, act_full, loss_full,
     size, layer, act, loss, dataset_name, full_name) = derive_name(cfg)
    log.info(f'Canonical name: {full_name}  (dataset={dataset_name})')

    out_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    pth_out = os.path.join(out_dir, full_name + '.pth')
    txt_out = os.path.join(out_dir, full_name + '.txt')

    if (os.path.exists(pth_out) or os.path.exists(txt_out)) and not args.force:
        log.error(f'Archive already exists at {pth_out} (and/or .txt). '
                  f'Pass --force to overwrite.')
        sys.exit(1)

    # Eval dataset: prefer valset, else build test split from trainset class.
    if 'valset' in cfg:
        eval_set = cfg['valset']
    elif 'trainset' in cfg:
        eval_set = type(cfg['trainset'])(train=False, center=True)
    else:
        log.error('Settings has neither valset nor trainset.')
        sys.exit(1)

    device = torch.device(args.device)
    model = cfg['model'].to(device)
    state = torch.load(state_dict_path, map_location=device)
    # Some layers (e.g. CPLConv2d) lazily register buffers on first forward;
    # run a dummy pass so load_state_dict finds those keys.
    from torch.utils.data import DataLoader
    _loader = DataLoader(eval_set, batch_size=2, shuffle=False)
    _x, _ = next(iter(_loader))
    model.train()
    with torch.no_grad():
        _ = model(_x.to(device))
    model.load_state_dict(state)
    model.eval()

    # Memory measurements first, while model state is fresh.
    log.info('Measuring inference memory...')
    mem_eval = measure_memory(model, eval_set, args.batch_size,
                              cfg['loss'], device, mode='eval', log=log)
    log.info('Measuring training memory (forward+backward, training batch_size)...')
    train_set = cfg.get('trainset')
    train_bs = cfg.get('batch_size', 256)
    mem_train = None
    if train_set is not None:
        mem_train = measure_memory(model, train_set, train_bs,
                                   cfg['loss'], device, mode='train', log=log)

    # train-mode pass above may have drifted parametrization buffers; reload.
    model.load_state_dict(state)
    model.eval()

    log.info(f'Running PGD eval at eps={args.eps}...')
    evaluator = EvaluatePGD(
        model=model, valset=eval_set, device=device,
        eps_list=args.eps, n_iter=args.n_iter,
        batch_size=args.batch_size, num_workers=2, logger=log,
    )
    pgd_summary = evaluator.run(save_path=None)

    training_stats = read_training_stats(source_dir, log)

    log.info(f'Copying state dict -> {pth_out}')
    shutil.copy2(state_dict_path, pth_out)

    archive_ts = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    write_summary(
        txt_out,
        full_name=full_name, dataset=dataset_name,
        source_run_dir=source_dir, source_state_dict=state_dict_path,
        archive_ts=archive_ts,
        size_full=size_full, layer_full=layer_full,
        act_full=act_full, loss_full=loss_full,
        cfg=cfg, training_stats=training_stats,
        memory_train=mem_train, memory_eval=mem_eval,
        pgd_summary=pgd_summary, eps_list=args.eps, args=args,
    )
    log.info(f'Wrote summary -> {txt_out}')


if __name__ == '__main__':
    main()
