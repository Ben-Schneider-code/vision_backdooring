import transformers
from torch.utils.data import DataLoader

from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.defenses.data_cleaning.spectre.rep_saver import rep_save
from src.model.model import Model
from src.defenses.data_cleaning.strip.STRIP import run

import os

from src.utils.distributed_validation import poison_validation_ds

my_model = None


def main(config_args: ConfigArgs):
    if config_args.exists():
        env_args: EnvArgs = config_args.get_env_args()
        backdoored_model_args: BackdooredModelArgs = config_args.get_backdoored_model_args()
        model_args: ModelArgs = config_args.get_model_args()
        dataset_args: DatasetArgs = config_args.get_dataset_args()
    else:
        print("Config not find")
        exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(env_args.gpus[0])
    model, backdoor = backdoored_model_args.unpickle(model_args, env_args)
    model: Model = model.cuda().eval()
    backdoor: Backdoor = backdoor

    # Compatibility hack
    backdoor.in_classes = None

    print(model.state_dict().keys())
    print(len(model.state_dict().keys()))
    print(list(model.state_dict().keys())[-1])
    # rep_save(model, ds, "placeholder", )
    global my_model
    my_model = model

def parse_args():
    parser = transformers.HfArgumentParser(ConfigArgs)
    return parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    main(*parse_args())
