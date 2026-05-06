# 1-Lipschitz CNN Compared

A benchmarking codebase for **1-Lipschitz convolutional neural network** under certifiable adversarial robustness. It is built on top of [`berndprach/1LipschitzLayersCompared`](https://github.com/berndprach/1LipschitzLayersCompared) (Prach, Brau, Buttazzo & Lampert, CVPR 2024), extended for my Cambridge Part III essay to include more activations / losses and an empirical (PGD) robustness evaluator.

<img src="https://github.com/DongshenWu/lcnn_essay/cover.png" alt="Radar plot of results" width="800"/>

A single training run is a 5-tuple **(dataset, model, layer, activation, loss)** plus optimiser hyperparameters, all expressed in a YAML settings file. Everything else — architecture wiring, certified-radius evaluation, hyperparameter sampling — is shared infrastructure.

## What you can compose

| Slot | Available choices | Abbrev. used in `results/` |
|------|-------------------|----------------------------|
| **Dataset** | `CIFAR10`, `CIFAR100`, `FashionMNIST`, `TinyImageNet` | — |
| **Model** | `ConvNet{XS,S,M,L}` (32×32 inputs) · `TConvNet{XS,S,M,L}` (64×64 inputs) | `XS, S, M, L` (or `TXS…`) |
| **1-Lipschitz layer** | `AOLConv2d{,Dirac,Orth}`, `BCOP`, `CayleyConv`, `CPLConv2d`, `LOT`, `SLLConv2d`, `SOC`, `SpectralNormConv2d{,Strict}`, `SandwichConv` | `AOL, BCOP, Cayley, CPL, LOT, SLL, SOC, SN, Sand` |
| **Activation** | `MaxMin`, `Abs`, `LinearSpline`, `Householder`, `NActivation` | `GS, AV, LS, HH, NA` |
| **Loss** | `LipCrossEntropyLoss`, `MulticlassHingeWithMargin` | `CE, HM` |

`SpectralNormConv2d` and the four extra activations (`Abs`, `LinearSpline`, `Householder`, `NActivation`) are fork additions — see [docs/SPEC.md](docs/SPEC.md). The seven canonical Lipschitz layers from the upstream paper are unchanged.

## Repository layout

```
lcnn/
├── src/                       # the engine (everything needed for a single run)
│   ├── train.py                  # entry point: settings.yml -> trained model
│   ├── parsers/                  # YAML tag resolver (!model, !layer, !activation, !metric, …)
│   ├── models/                   # ConvNet + Lipschitz layers + activations
│   ├── datasets/                 # CIFAR10/100, FashionMNIST, TinyImageNet loaders
│   ├── trainer/                  # training loop, optimiser, lr_scheduler, metrics, losses
│   ├── evaluator/                # certified RA, L2 PGD, memory, batch-time
│   └── utils/                    # statistics, inspections, line formatting, results readers
├── tools/                     # multi-run orchestration & post-processing (not on the run path)
│   ├── make_tree.py              # generate a settings tree for random search / final training
│   ├── epoch_budget_estimator.py # calibrate per-(model, layer) epoch counts to a wall-clock budget
│   ├── eval_best_lr_wd.py        # pick best (lr, wd) per cell from a search tree
│   ├── pgd_eval.py               # post-training empirical RA (L2 PGD)
│   ├── archive_to_results.py     # canonical archival -> results/<DATASET>/<name>.{txt,pth}
│   ├── run.py                    # dispatcher for memory/throughput measurements
│   ├── train_model.py, measure_{memories,batch_times}.py
│   ├── grid_make_settings.py
│   └── plot/                     # all plot_*.py scripts
├── tests/                     # unittest suite
└── settings/                  # default YAMLs (one per dataset)
```

A directory `data/` is created at runtime for per-run workspace, dataset cache, and measurements:
```
data/runs/<DATASET>/<experiment>/<ConvNet*>/<Layer>/<timestamp.jobid>/
    settings.yml             # the input
    dumped_setting.yml       # the input with random samples resolved
    model_state_dict.pth
    training_statistics.csv  # per-epoch metrics
    training.log
data/datasets/               # torchvision-cached raw data
data/settings/<DATASET>/     # epoch_budget.yml, best_lr_wd.csv
data/evaluations/            # memory.csv, batch_times.csv
```

## How a run flows

```
        settings.yml
              │
              ▼
   ┌────────────────────┐
   │  src/parsers/      │   resolves !model / !layer / !activation /
   │  SettingParser     │   !dataset / !optimizer / !lr_scheduler /
   │                    │   !metric and samples !choice / !randint /
   │                    │   !randfloat / !randlog10 once
   └────────────────────┘
              │
              ▼
   model · trainset · valset · optimizer · lr_scheduler · loss · eval_metrics
              │
              ▼
   ┌────────────────────┐
   │  src/trainer/      │   per-batch fwd+bwd, per-epoch eval,
   │  Train.run()       │   writes training.log + training_statistics.csv,
   │                    │   saves model_state_dict.pth
   └────────────────────┘
              │
              ▼
   run_dir/  ─────────────►  tools/pgd_eval.py  (empirical robustness)
                             tools/archive_to_results.py  (canonical metrics + summary)
```

## Setup

Run everything from the **repo root** (paths in YAML and code are CWD-relative).

```bash
# torch first; pick the cuda build matching your GPU
pip install torch torchvision

# the rest
pip install einops==0.5.0 matplotlib==3.7.2 numpy pandas PyYAML==6.0 tqdm==4.64.0
pip install torchattacks   # for tools/pgd_eval.py
```

`requirements.txt` pins `torch==1.12`; the essay was actually run on `torch 2.8` / RTX 4090. Both work.

## Running an experiment

The YAML settings file is the only source of truth for a run. Below is a minimal complete example: **FashionMNIST, ConvNetXS, AOL conv, LinearSpline activation, hinge-with-margin loss**.

Save the following as `data/runs/FashionMNIST/my_run/ConvNetXS/AOL_LS_HM/settings.yml`:

```yaml
batch_size: 256
num_workers: 4
epochs: 100

model: !model
  name: SimplifiedConvNet
  model_id: ConvNetXS
  get_conv: !layer AOLConv2dDirac
  get_conv_first: !layer AOLConv2dOrth
  get_conv_head:  !layer AOLConv2dOrth
  get_activation: !activation
    name: LinearSpline
    num_knots: 21
    range: 3.0
    init: absolute_value
  nrof_classes: 10

trainset: !dataset
  name: FashionMNIST
  train: True
  center: True

valset: !dataset
  name: FashionMNIST
  train: False
  center: True

optimizer: !optimizer
  name: SGD
  lr: 0.1
  weight_decay: 1.0e-5
  momentum: 0.9

lr_scheduler: !lr_scheduler
  name: OneCycleLR

loss: !metric
  name: MulticlassHingeWithMargin
  margin: 0.1414        # = sqrt(2) * 36/255 in logit units

eval_metrics:
  accuracy:    !metric { name: Accuracy }
  robstacc_36:  !metric { name: RobustAccuracy, eps: 0.1411 }
  robstacc_72:  !metric { name: RobustAccuracy, eps: 0.2824 }
  robstacc_108: !metric { name: RobustAccuracy, eps: 0.4235 }
  robstacc_255: !metric { name: RobustAccuracy, eps: 1.0 }
  throughput:  !metric { name: Throughput }
```

### Train

```bash
python3 src/train.py data/runs/FashionMNIST/my_run/ConvNetXS/AOL_LS_HM \
    --device cuda --jobid test
```

`src/train.py` creates a timestamped subdir inside the run directory, dumps the resolved YAML to `dumped_setting.yml`, and starts training. Add `--debug` for a one-epoch run with extra instrumentation; add `--save-memory` to skip writing the state dict.

### What you get back

In the timestamped subdir:
- `training_statistics.csv` — per-epoch `train_*`, `val_*` columns including `val_accuracy`, `val_robstacc_{36,72,108,255}`, `val_throughput`.
- `model_state_dict.pth` — final weights (unless `--save-memory`).
- `dumped_setting.yml` — the exact resolved settings (with random search values realised).
- `training.log` — readable log.

### Random search → final training

Search over (lr, weight_decay) per (model, layer) cell, then re-train the best:

```bash
# 1. Calibrate epoch budget for a 2-hour wall-clock target per cell.
python3 tools/epoch_budget_estimator.py --dataset CIFAR10
# writes data/settings/CIFAR10/epoch_budget.yml

# 2. Generate a settings tree of random-search runs.
python3 tools/make_tree.py \
    --root_dir data/runs/CIFAR10/random_search \
    --default settings/defaults_CIFAR10.yml \
    --mode random_search \
    --training-time 2

# 3. Train each leaf however many seeds you want.
for cell in data/runs/CIFAR10/random_search/*/*/; do
    python3 src/train.py "$cell" --device cuda --jobid 0
done

# 4. Pick the best (lr, wd) per cell.
python3 tools/eval_best_lr_wd.py \
    --runs-path data/runs/CIFAR10/random_search \
    --output-path data/settings/CIFAR10/best_lr_wd.csv

# 5. Generate final-training tree using those best hyperparameters.
python3 tools/make_tree.py \
    --root_dir data/runs/CIFAR10/final_runs \
    --default settings/defaults_CIFAR10.yml \
    --mode final_training \
    --training-time 8 \
    --best-lr-wd-file data/settings/CIFAR10/best_lr_wd.csv

# 6. Train the final cells.
for cell in data/runs/CIFAR10/final_runs/*/*/; do
    python3 src/train.py "$cell" --device cuda --jobid 0
done
```

## Evaluating a trained run

CRA (certified robust accuracy) is already in `training_statistics.csv` — every epoch reports it at the four standard radii {36, 72, 108, 255}/255.

For **empirical** robust accuracy (L2 PGD attack):

```bash
python3 tools/pgd_eval.py data/runs/FashionMNIST/my_run/ConvNetXS/AOL_LS_HM/<timestamp> \
    --device cuda --eps 0.1411 0.2824 0.4235 1.0 --n-iter 20 --seed 0
```

Writes `pgd_metrics.csv` next to the state dict. Reports both certified and empirical RA per radius — the soundness invariant `CRA ≤ emp_RA ≤ clean` is self-checking.

For a **single canonical summary** of a finished run (clean acc, CRA + emp-RA at four radii, wall-clock, throughput, peak memory):

```bash
python3 tools/archive_to_results.py data/runs/FashionMNIST/my_run/ConvNetXS/AOL_LS_HM
```

This re-runs PGD, measures memory & throughput, and writes
`results/FashionMNIST/XS_AOL_LS_HM.txt` (tracked) and `.pth` (gitignored). The file naming is `<size>_<layer>_<act>_<loss>` using the abbreviations in the table at the top of this README.

## Memory and throughput measurement (independent of training)

```bash
# List the (model, layer) combinations to job-ids
python3 tools/run.py --task=print-job-id-assignment

# Measure peak GPU memory for one combination
python3 tools/run.py --job-id=0 --task=measure-memory \
    --dataset=cifar10 --results-file-name=memory.csv

# Measure batch time
python3 tools/run.py --job-id=0 --task=measure-batch-times \
    --dataset=cifar10 --results-file-name=batch_times.csv
```

Outputs land in `data/evaluations/<results-file-name>`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Run a single module:

```bash
python3 -m unittest tests.test_layers.test_householder -v
```

`tests/__init__.py` puts `src/` on the import path, so tests run from the repo root with no extra setup.
