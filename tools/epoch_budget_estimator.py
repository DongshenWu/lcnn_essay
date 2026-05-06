import os
import sys
from argparse import ArgumentParser
from functools import partial
from time import time
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import SGD
from torch.utils.data import DataLoader

import datasets
from models import layers
from models.simplified_conv_net import DEFAULT_MODELS, conv_net


def epoch_budget_estimator(
        model: torch.nn.Module,
        device: torch.device,
        training_time: float,
        data_loader: torch.utils.data.DataLoader,
        num_iters: int
):
    """How many full epochs fit in `training_time` hours, measured by per-iter wall clock."""
    model.train()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    delta_time = []
    prev_time = None
    while len(delta_time) < num_iters:
        for x, l in data_loader:
            x, l = x.to(device), l.to(device)
            out = model(x)
            loss = loss_fn(out, l)
            loss.backward()
            optimizer.step()
            # First iter is warm-up — start the clock on the second.
            if prev_time is None:
                prev_time = time()
            else:
                actual_time = time()
                delta_time.append(actual_time - prev_time)
                prev_time = actual_time

    assert not torch.isnan(out).any(), 'Nan value in the output'
    eta = np.mean(delta_time[-num_iters//2:])
    return int(training_time * 3600 / (eta * len(data_loader)))


def update_layers_dict(old_dict: Dict[str, dict], new_dict: Dict[str, dict]):
    for key, value in new_dict.items():
        if key in old_dict.keys():
            old_dict[key] = {**old_dict[key], **value}
        else:
            old_dict[key] = value
    return old_dict


def parse_arguments():
    parser = ArgumentParser(
        description='Calibrate per-(model, layer) epoch counts to a wall-clock training budget.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--training_time', type=float, default=2, help='hours')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet',
                                 'FashionMNIST'])
    parser.add_argument('--debug', action='store_true',
                        help='2 iters per config (smoke).')
    parser.add_argument('--update-layers', type=str, nargs='+', default=None,
                        help='Re-measure only these layers, merging into existing budget.')
    return parser.parse_args()


def main():

    args = parse_arguments()
    num_iters = 2 if args.debug else 100
    dataset = getattr(datasets, args.dataset)(train=True)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    if args.update_layers is not None:
        conv_layers_names = args.update_layers
    else:
        conv_layers_names = layers.available_conv2d_layers()
    conv_layers = list(map(partial(getattr, layers), conv_layers_names))
    if args.dataset == 'CIFAR10':
        default_models = ['ConvNetXS', 'ConvNetS', 'ConvNetM', 'ConvNetL']
        nrof_classes = 10
    elif args.dataset == 'CIFAR100':
        default_models = ['ConvNetXS', 'ConvNetS', 'ConvNetM', 'ConvNetL']
        nrof_classes = 100
    elif args.dataset == 'FashionMNIST':
        default_models = ['ConvNetXS', 'ConvNetS', 'ConvNetM', 'ConvNetL']
        nrof_classes = 10
    elif args.dataset == 'TinyImageNet':
        default_models = ['TConvNetXS', 'TConvNetS', 'TConvNetM', 'TConvNetL']
        nrof_classes = 200
    df = pd.DataFrame(columns=['Model ID', 'Method', 'Epochs'])
    for model_name in default_models:
        print(f'{model_name}')
        for conv_layer in conv_layers:
            print(f'\t{conv_layer.__name__}')
            linear = None
            if conv_layer.__name__ == 'BCOP':
                linear = layers.BjorckLinear
            elif conv_layer.__name__ == 'CayleyConv':
                linear = layers.CayleyLinear
            elif conv_layer.__name__ == 'SandwichConv':
                linear = layers.SandwichFc

            try:
                net = conv_net(model_name,
                               get_conv=conv_layer,
                               get_activation=layers.MaxMin,
                               get_linear=linear,
                               nrof_classes=nrof_classes)
                epochs = epoch_budget_estimator(
                    model=net, device=args.device,
                    training_time=args.training_time,
                    data_loader=data_loader,
                    num_iters=num_iters)

                row = {'Model ID': model_name,
                       'Method': conv_layer.__name__,
                       'Epochs': epochs}
                df = pd.concat([df, pd.DataFrame(row, index=[0])],
                               ignore_index=True)
            except RuntimeError as error:
                print(
                    f'Runtime Error: {error} for {model_name}, {conv_layer.__name__}')
                epochs = 0
            except AssertionError as error:
                print(
                    f'Assertion Error: {error} for {model_name}, {conv_layer.__name__}')
                epochs = 0

            if args.debug:
                print('Debug Mode. the test ends here')
                break
        print(df)

    df.reset_index(drop=True, inplace=True)
    df.set_index(['Method'], inplace=True)
    df_pivoted = df.pivot(columns='Model ID', values='Epochs')
    epochs_dict = df_pivoted.to_dict()
    epoch_budget_dname = f'data/settings/{args.dataset}'
    epoch_budget_fname = os.path.join(epoch_budget_dname, 'epoch_budget.yml')
    if args.update_layers is not None:
        with open(epoch_budget_fname, 'r') as f:
            epochs_dict = update_layers_dict(
                yaml.load(f, Loader=yaml.SafeLoader), epochs_dict)
    else:
        os.makedirs(epoch_budget_dname, exist_ok=True)
    with open(epoch_budget_fname, 'w') as f:
        f.write(f'# {torch.cuda.get_device_name()} | {args.dataset}\n')
        f.write(f'# batch_size={args.batch_size}, training_time={args.training_time}h\n\n')
        yaml.dump(epochs_dict, f)


if __name__ == '__main__':
    main()
