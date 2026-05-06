import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import datasets

from models import get_model
from evaluator.memory_usage import get_model_memory

SEPERATOR = ";"


def measure_memories(idx: int,
                     model_name: str,
                     layer_name: str,
                     results_file_name: str,
                     dataset_cls: datasets.DatasetClass):

    def load_model():
        return get_model(model_name, layer_name)

    if results_file_name is None:
        results_file_name = "measured_memories.csv"

    train_loader, test_loader = datasets.get_train_test_loader(dataset_cls)

    results_folder = os.path.join("data", "evaluations")
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, results_file_name)

    exception = None
    exception_str = str(None)
    result_dict = None

    # First job in a sweep: separator line in the shared CSV.
    if idx == 0:
        with open(results_file, "a") as f:
            f.write("=" * 30 + "\n")

    try:
        result_dict = get_model_memory(load_model, test_loader, train_loader)
    except Exception as e:
        print(f"Error while evaluating "
              f"model {model_name} and layer {layer_name}!\n "
              f"{e}")
        exception = e
        exception_str = str(e).replace(SEPERATOR, ",")

    with open(results_file, "a") as f:
        sep = SEPERATOR
        f.write(f"{idx}{sep} "
                f"{model_name}{sep} "
                f"{layer_name}{sep} "
                f"{result_dict}{sep} "
                f"{exception_str}\n")

    if exception is not None:
        raise exception

