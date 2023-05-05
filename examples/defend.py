import numpy as np
import transformers

from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.defenses.defense import Defense
from src.defenses.defense_factory import DefenseFactory
from src.model.model import Model
from src.observers.baseobserver import BaseObserver
from src.observers.observer_factory import ObserverFactory
from src.utils.special_print import print_dict_highlighted


def parse_args():
    parser = transformers.HfArgumentParser((EnvArgs,
                                            BackdooredModelArgs,
                                            DefenseArgs,
                                            DatasetArgs,
                                            ObserverArgs,
                                            OutdirArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def defend(env_args: EnvArgs,
                     backdoored_model_args: BackdooredModelArgs,
                     defense_args: DefenseArgs,
                     dataset_args: DatasetArgs,
                     observer_args: ObserverArgs,
                     outdir_args: OutdirArgs,
                     config_args: ConfigArgs):
    """ Run a defense against a backdoor. """
    if config_args.exists():
        env_args = config_args.get_env_args()
        defense_args = config_args.get_defense_args()
        backdoored_model_args = config_args.get_backdoored_model_args()
        dataset_args = config_args.get_dataset_args()
        observer_args = config_args.get_observer_args()
        outdir_args = config_args.get_outdir_args()

    backdoor: Backdoor = backdoored_model_args.load_backdoor(env_args=env_args).eval()
    model: Model = backdoored_model_args.load_model(env_args=env_args)

    print_dict_highlighted(vars(backdoor.backdoor_args))

    print_dict_highlighted(vars(model.model_args))
    print_dict_highlighted(vars(backdoored_model_args.backdoor_args))

    defense: Defense = DefenseFactory.from_defense_args(defense_args, env_args=env_args)
    observers = ObserverFactory.from_observer_args(observer_args, env_args=env_args)
    defense.add_observers(observers)

    ## Build all datasets for evaluating the defense.
    # 0. The clean training data
    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)

    # 1. The clean test dataset
    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    #ds_test = ds_test.random_subset(10_000)

    backdoor.visualize(ds_test)

    # 2. The poisoned test dataset w/o target class (used to measure ASR)
    ds_poison_asr = ds_test.remove_classes([backdoor.backdoor_args.target_class]) \
        .random_subset(500).set_poison_label(backdoor.backdoor_args.target_class).add_poison(backdoor=backdoor,
                                                                                             poison_all=True)

    # 3. The poisoned test dataset w/o the target class but with oracle labels (used to measure ARR)
    ds_poison_arr = ds_test.remove_classes([backdoor.backdoor_args.target_class]) \
        .random_subset(500).set_poison_label(False).add_poison(backdoor=backdoor, poison_all=True)

    # print(f"ARR Before: {model.evaluate(ds_poison_arr):.4f}")

    # Take 10 random classes
    rnd_classes = np.arange(len(ds_train.classes))
    np.random.shuffle(rnd_classes)
    classes_to_remove = rnd_classes[50:].tolist()
    if 0 in classes_to_remove:
        classes_to_remove.remove(0)  # keep the target class
    subset = ds_test.without_transform()
    subset = subset.random_subset(5_000).remove_classes(classes_to_remove, verbose=False)
    subset_psn = subset.remove_classes([backdoor.backdoor_args.target_class]) \
        .random_subset(50).set_poison_label(1001).add_poison(backdoor=backdoor, poison_all=True)
    #subset_psn.visualize(3)
    to_plot = subset.concat(subset_psn).without_transform()
    asr = model.evaluate(ds_poison_asr, verbose=False)
    cda = model.evaluate(ds_test, verbose=False)

    model.debug(True)
    model.get_saliency_map(*ds_poison_asr[0])
    print(f"CDA Before: {cda:.4f}, asr before {asr:.4f}")
    clean_model = defense.apply(model, to_plot=to_plot, ds_train=ds_train, ds_poison_arr=ds_poison_arr, backdoor=backdoor,
                                ds_poison_asr=ds_poison_asr, ds_test=ds_test)

    print(f"CDA After: {clean_model.evaluate(ds_test):.4f}")
    print(f"ASR After: {clean_model.evaluate(ds_poison_asr):.4f}")
    print(f"ARR After: {clean_model.evaluate(ds_poison_arr):.4f}")

    observer: BaseObserver
    for observer in observers:
        observer.plot()
        observer.attach_config(DefenseArgs.CONFIG_KEY, defense_args)
        observer.attach_config(backdoored_model_args.get_backdoor_args().CONFIG_KEY, backdoored_model_args.get_backdoor_args())
        observer.attach_config(DatasetArgs.CONFIG_KEY, dataset_args)
        observer.save(outdir_args)

if __name__ == "__main__":
    defend(*parse_args())
