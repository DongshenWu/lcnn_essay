#!/usr/bin/env python3
"""Fan out the MaxMin baseline tree from make_tree.py to 4 sibling activation trees.

Reads the GS tree at `data/runs/<dataset>/grid_final_GS/ConvNetXS/{AOL,SOC,CPL}/`
and writes `grid_final_{AV,HH,LS,NA}/...`. Also rewrites `weight_decay: 1e-05`
to `1.0e-05` (PyYAML SafeLoader rejects dot-less mantissas).
"""
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANT_ACTIVATION = {
    'GS': '  get_activation: !activation MaxMin\n',
    'AV': '  get_activation: !activation Abs\n',
    'HH': '  get_activation: !activation { name: Householder }\n',
    'LS': ('  get_activation: !activation { name: LinearSpline, '
           'num_knots: 21, range: 3.0, init: absolute_value }\n'),
    'NA': ('  get_activation: !activation { name: NActivation, '
           'init: absid, lr_scale: 0.1 }\n'),
}

ACTIVATION_LINE_RE = re.compile(
    r'^\s*get_activation:\s*!(?:layer|activation)\s+MaxMin\s*$')
WD_FIX_RE = re.compile(r'^(\s*weight_decay:\s*)(\d+)(e-\d+)\s*$')


def fix_wd_line(line: str) -> str:
    m = WD_FIX_RE.match(line)
    if m:
        return f'{m.group(1)}{m.group(2)}.0{m.group(3)}\n'
    return line


def transform_settings(src_path: str, dst_path: str, activation_line: str):
    with open(src_path) as f:
        lines = f.readlines()
    out = []
    replaced = False
    for line in lines:
        if ACTIVATION_LINE_RE.match(line):
            out.append(activation_line)
            replaced = True
        else:
            out.append(fix_wd_line(line))
    if not replaced:
        raise RuntimeError(
            f'Did not find `get_activation: !layer MaxMin` in {src_path}; '
            f'cannot apply variant transformation.')
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w') as f:
        f.writelines(out)


def patch_wd_only(path: str):
    with open(path) as f:
        lines = f.readlines()
    out = [fix_wd_line(line) for line in lines]
    with open(path, 'w') as f:
        f.writelines(out)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='FashionMNIST')
    args = p.parse_args()

    grid_base = os.path.join(REPO_ROOT, 'data/runs', args.dataset)
    source_tree = os.path.join(grid_base, 'grid_final_GS')

    if not os.path.isdir(source_tree):
        print(f'Error: source tree {source_tree} does not exist. '
              f'Run make_tree.py first.', file=sys.stderr)
        sys.exit(1)

    target_layers = ['AOLConv2d', 'SOC', 'CPLConv2d']
    src_files = []
    for layer in target_layers:
        src = os.path.join(source_tree, 'ConvNetXS', layer, 'settings.yml')
        if not os.path.isfile(src):
            print(f'Warning: missing {src}', file=sys.stderr)
            continue
        src_files.append((layer, src))

    if not src_files:
        print('No source settings.yml files found; nothing to do.',
              file=sys.stderr)
        sys.exit(1)

    for _, src in src_files:
        patch_wd_only(src)
    # Rewrite GS in place so its tag matches the siblings (!activation, not !layer).
    for layer, src in src_files:
        transform_settings(src, src, VARIANT_ACTIVATION['GS'])

    for variant in ('AV', 'HH', 'LS', 'NA'):
        dst_root = os.path.join(grid_base, f'grid_final_{variant}')
        for layer, src in src_files:
            dst = os.path.join(dst_root, 'ConvNetXS', layer, 'settings.yml')
            transform_settings(src, dst, VARIANT_ACTIVATION[variant])
            print(f'Wrote {dst}')

    print(f'\nDone. 5 trees x 3 layers = 15 settings.yml files generated under '
          f'{grid_base}/grid_final_{{GS,AV,HH,LS,NA}}/ConvNetXS/.')


if __name__ == '__main__':
    main()
