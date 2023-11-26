from typing import List

from torch import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

import torch
import transformers


from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.arguments.grid_search_args import GridSearchArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.defenses.defense_factory import DefenseFactory
from src.model.model import Model
from src.observers.baseobserver import BaseObserver
from src.observers.observer_factory import ObserverFactory
from src.utils.special_print import print_dict_highlighted, print_highlighted, print_warning


def parse_args():
    parser = transformers.HfArgumentParser((EnvArgs,
                                            BackdooredModelArgs,
                                            GridSearchArgs,
                                            DatasetArgs,
                                            ObserverArgs,
                                            OutdirArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()

def _search(valid_params, backdoored_model_args, env_args, dataset_args, grid_search_args,
                                                          observer_args, gpu_id, shared_mem: dict):
    """ Process for multiprocessing.
    """
    torch.cuda.set_device(gpu_id)
    backdoor: Backdoor = backdoored_model_args.load_backdoor(env_args=env_args)

    # Clean Data
    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)
    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False).random_subset(5_000)

    # Poisoned data_cleaning
    ds_poison_asr = ds_test.remove_classes([backdoor.backdoor_args.target_class]) \
        .random_subset(500).set_poison_label(backdoor.backdoor_args.target_class).add_poison(backdoor=backdoor,
                                                                                             poison_all=True)
    observers = ObserverFactory.from_observer_args(observer_args, grid_search_args=grid_search_args, env_args=env_args)

    model: Model = backdoored_model_args.load_model(env_args=env_args)
    print_highlighted(f"Found {len(valid_params)} set of parameters!")
    for i, defense_args in enumerate(valid_params):
        print()
        print(f"({i + 1}/{len(valid_params)}): {defense_args}")
        defense = DefenseFactory.from_defense_args(defense_args, env_args=env_args)
        defense.add_observers(observers)
        for observer in observers:
            observer.attach_config(defense.get_id(), defense_args)
            observer.attach_config("defense_args", defense_args)
        clean_model = defense.apply(model.deepcopy(), ds_train=ds_train, ds_test=ds_test, ds_poison_asr=ds_poison_asr)

        print(f"({i + 1}/{len(valid_params)}): CDA After: {clean_model.evaluate(ds_test):.4f}")
        print(f"({i + 1}/{len(valid_params)}): ASR After: {clean_model.evaluate(ds_poison_asr):.4f}")
        print()

def grid_search(env_args: EnvArgs,
                backdoored_model_args: BackdooredModelArgs,
                grid_search_args: GridSearchArgs,
                dataset_args: DatasetArgs,
                observer_args: ObserverArgs,
                outdir_args: OutdirArgs,
                config_args: ConfigArgs):
    """ Performs a grid search given one backdoor model and a '*.yml' file of allowed parameters.
        This file specifies all parameter combinations for a defense to be ablated over.
    """
    print_warning(f"This function does not yet support data recording!")
    if config_args.exists():
        env_args = config_args.get_env_args()
        backdoored_model_args = config_args.get_backdoored_model_args()
        dataset_args = config_args.get_dataset_args()
        observer_args = config_args.get_observer_args()
        outdir_args = config_args.get_outdir_args()
        grid_search_args: GridSearchArgs = config_args.get_grid_search_args()

    valid_params: List[DefenseArgs] = grid_search_args.valid_parameters()

    # Set the 'spawn' start method for multiprocessing
    # Start the multi gpu loop

    manager = multiprocessing.Manager()
    shared_results = manager.dict()

    chunk_size = int(len(valid_params) / env_args.num_gpus)
    valid_params_chunks = [valid_params[i*chunk_size:(i+1)*chunk_size] for i in range(int(len(valid_params) / chunk_size))]
    processes = []
    for i in range(env_args.num_gpus):
        p = multiprocessing.Process(target=_search, args=(valid_params_chunks[i],
                                                          backdoored_model_args,
                                                          env_args,
                                                          dataset_args,
                                                          grid_search_args,
                                                          observer_args, i, shared_results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    grid_search(*parse_args())
