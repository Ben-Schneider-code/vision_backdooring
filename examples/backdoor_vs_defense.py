from dataclasses import asdict

import transformers

from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.defenses.defense import Defense
from src.defenses.defense_factory import DefenseFactory
from src.model.model import localize, Model
from src.utils.distributed_validation import poison_validation_ds
from src.utils.special_print import print_dict_highlighted, print_highlighted


def main(config_args: ConfigArgs):
    if config_args.exists():
        env_args = config_args.get_env_args()
        backdoored_model_args = config_args.get_backdoored_model_args()
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        defense_args = config_args.get_defense_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(env_args.gpus[0])

    model, backdoor = backdoored_model_args.unpickle(model_args, env_args)

    # restore feature recording after unpickling
    model.add_features_hook = Model.add_features_hook
    model.activate_feature_recording = Model.activate_feature_recording

    model.eval()

    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)

    ds_val: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_poisoned: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_poisoned = poison_validation_ds(ds_poisoned, backdoor, len(ds_poisoned))

    print(model.evaluate(ds_val, verbose=True))
    print(model.evaluate(ds_poisoned, verbose=True))

    defense: Defense = DefenseFactory.from_defense_args(defense_args, env_args=env_args)
    print_highlighted(defense.defense_args.def_name)
    defense.apply(model, ds_train, backdoor=backdoor, ds_test=ds_val, ds_poison_asr=ds_poisoned)


    ds_val: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_poisoned: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_poisoned = poison_validation_ds(ds_poisoned, backdoor, len(ds_poisoned))

    print(model.evaluate(ds_val, verbose=True))
    print(model.evaluate(ds_poisoned, verbose=True))


def parse_args():
    parser = transformers.HfArgumentParser(ConfigArgs)
    return parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    main(*parse_args())
