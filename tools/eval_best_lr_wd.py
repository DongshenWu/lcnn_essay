import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from utils.results import get_statistics, get_best_lr_wd

parser = ArgumentParser('Pick the best (lr, wd) per (model_id, get_conv) cell from a random-search tree.')
parser.add_argument('--runs-path', type=str, default='.')
parser.add_argument('--output-path', default='./best_lr_wd.csv', type=str)
parser.add_argument('--all-runs', action='store_true',
                    help='Include incomplete runs.')

HYPER_PARAMETERS = ['model_id', 'get_conv', 'lr', 'weight_decay', 'epochs']
METRICS = ['val_accuracy', 'val_robstacc_36', 'epoch']


def main():
    args = parser.parse_args()
    df_stats = get_statistics(
        args.runs_path, metrics=METRICS, hyper_params=HYPER_PARAMETERS,
        mode='random_search', only_completed=not args.all_runs,
    )
    df_best = get_best_lr_wd(df_stats)
    print(df_best)
    df_best.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
