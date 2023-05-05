import numpy as np
from torch import multiprocessing

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.model_args import ModelArgs
from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor_factory import BackdoorFactory
from src.model.model_factory import ModelFactory
from src.trainer.trainer import Trainer
from src.trainer.trainer_factory import TrainerFactory

if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

import torch
import transformers

from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.grid_search_args import GridSearchArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.model.model import Model
from src.utils.special_print import print_dict_highlighted


def parse_args():
    parser = transformers.HfArgumentParser((EnvArgs,
                                            BackdooredModelArgs,
                                            GridSearchArgs,
                                            DatasetArgs,
                                            ObserverArgs,
                                            OutdirArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def _embed(model_args: ModelArgs,
           backdoor_args: BackdoorArgs,
           env_args: EnvArgs,
           trainer_args: TrainerArgs,
           outdir_args: OutdirArgs,
           dataset_args: DatasetArgs,
           gpu_id: int):
    """ Process for multiprocessing.
    """
    torch.cuda.set_device(gpu_id)

    reps = int(np.ceil(env_args.experiment_repetitions / len(env_args.gpus)))
    folder_names = []
    for i in range(reps):
        outdir_args = OutdirArgs(outdir_args.root, outdir_args.name)
        ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)
        ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
        model: Model = ModelFactory.from_model_args(model_args, env_args=env_args)
        backdoor = BackdoorFactory.from_backdoor_args(backdoor_args, env_args=env_args)
        backdoor.before_attack(ds_train)

        if gpu_id == env_args.gpus[0]:
            backdoor.visualize(ds_test)

        ds_train = ds_train.add_poison(backdoor,
                                       boost=trainer_args.boost)  # let backdoor select poisoned images and apply poisoning
        trainer: Trainer = TrainerFactory.from_trainer_args(trainer_args, env_args=env_args)
        ds_poison = ds_test.remove_classes([backdoor.backdoor_args.target_class]).random_subset(2_000) \
            .add_poison(backdoor, poison_all=True).set_poison_label(backdoor.backdoor_args.target_class)

        backdoor.train()
        trainer.train(model=model, ds_poison=ds_poison, ds_train=ds_train, ds_test=ds_test.random_subset(5_000),
                      backdoor=backdoor, outdir_args=outdir_args)
        folder_names += [outdir_args.create_folder_name()]
    print(f"Process {gpu_id} finished. Folders: {folder_names}")


def grid_embed(outdir_args: OutdirArgs,
               env_args: EnvArgs,
               model_args: ModelArgs,
               trainer_args: TrainerArgs,
               dataset_args: DatasetArgs,
               backdoor_args: BackdoorArgs,
               config_args: ConfigArgs):
    """ Grid embeds a backdoor
    """
    if config_args.exists():
        outdir_args = config_args.get_outdir_args()
        env_args = config_args.get_env_args()
        model_args = config_args.get_model_args()
        trainer_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        backdoor_args = config_args.get_backdoor_args()

    print_dict_highlighted(vars(backdoor_args))
    processes = []
    for i in env_args.gpus:
        p = multiprocessing.Process(target=_embed, args=(model_args,
                                                         backdoor_args,
                                                         env_args,
                                                         trainer_args,
                                                         outdir_args,
                                                         dataset_args,
                                                         i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    grid_embed(*parse_args())
