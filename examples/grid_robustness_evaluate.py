from typing import List

import transformers

from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.arguments.grid_evaluate_args import GridEvaluateArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.defenses.defense_factory import DefenseFactory
from src.model.model import Model
from src.observers.baseobserver import BaseObserver
from src.observers.observer_factory import ObserverFactory
from src.utils.special_print import print_highlighted


def parse_args():
    parser = transformers.HfArgumentParser((GridEvaluateArgs,
                                            DatasetArgs,
                                            ObserverArgs,
                                            BackdooredModelArgs,
                                            OutdirArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def grid_evaluate(grid_evaluate_args: GridEvaluateArgs,
                  dataset_args: DatasetArgs,
                  observer_args: ObserverArgs,
                  backdoored_model_args: BackdooredModelArgs,
                  outdir_args: OutdirArgs,
                  env_args: EnvArgs,
                  config_args: ConfigArgs):
    """ Performs a grid search for a given attack against a set of defenses.
    """
    if config_args.exists():
        env_args = config_args.get_env_args()
        dataset_args = config_args.get_dataset_args()
        observer_args = config_args.get_observer_args()
        backdoored_model_args = config_args.get_backdoored_model_args()
        outdir_args = config_args.get_outdir_args()
        grid_evaluate_args: GridEvaluateArgs = config_args.get_grid_evaluate_args()

    # Clean Data
    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)
    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False).random_subset(10_000)

    backdoor: Backdoor = backdoored_model_args.load_backdoor(env_args=env_args)
    model: Model = backdoored_model_args.load_model(env_args=env_args)

    ds_poison_asr = ds_test.remove_classes([backdoor.backdoor_args.target_class]) \
        .random_subset(2000).set_poison_label(backdoor.backdoor_args.target_class).add_poison(backdoor=backdoor,
                                                                                              poison_all=True)

    ds_poison_asr.visualize(3)

    observers = ObserverFactory.from_observer_args(observer_args, grid_evaluate_args=grid_evaluate_args, env_args=env_args)
    all_defenses: List[DefenseArgs] = grid_evaluate_args.valid_parameters()
    print_highlighted(f"Found {len(all_defenses)} defenses to evaluate")
    for defense_args in all_defenses:
        print(f"{backdoor.backdoor_args.backdoor_name} vs {defense_args.def_name}")
        print(f"CDA Before: {model.evaluate(ds_test):.4f}")
        print(f"ASR Before: {model.evaluate(ds_poison_asr):.4f}")

        defense = DefenseFactory.from_defense_args(defense_args, env_args=env_args)
        defense.add_observers(observers)
        for observer in observers:
            observer.attach_config(defense.get_id(), defense_args)
            observer.attach_config("defense_args", defense_args)
            observer.plot()

        defended_model = defense.apply(model.deepcopy(), ds_train=ds_train, ds_test=ds_test,
                                       ds_poison_asr=ds_poison_asr, backdoor=backdoor)

        for observer in observers:
            observer.plot()
            observer.save(outdir_args)

        print(f"CDA After: {defended_model.evaluate(ds_test):.4f}")
        print(f"ASR After: {defended_model.evaluate(ds_poison_asr):.4f}")
        print()

    observer: BaseObserver
    for observer in observers:
        observer.plot()
        observer.save(outdir_args)


if __name__ == "__main__":
    grid_evaluate(*parse_args())
