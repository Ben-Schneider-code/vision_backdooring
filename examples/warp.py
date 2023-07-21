from copy import deepcopy
from typing import Tuple
from torch import nn
from torch.utils.data import DataLoader

from src.backdoor.backdoor import CleanLabelBackdoor
import torch
import torch.multiprocessing as mp
import transformers
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.trainer_args import TrainerArgs
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.utils.WarpGrid import WarpGrid
from src.utils.special_images import plot_images

mp.set_sharing_strategy('file_system')
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            BackdoorArgs,
                                            TrainerArgs,
                                            DatasetArgs,
                                            EnvArgs,
                                            OutdirArgs,
                                            ConfigArgs
                                            ))
    return parser.parse_args_into_dataclasses()


def _embed(model_args: ModelArgs,
           backdoor_args: BackdoorArgs,
           trainer_args: TrainerArgs,
           dataset_args: DatasetArgs,
           out_args: OutdirArgs,
           env_args: EnvArgs,
           config_args: ConfigArgs):
    if config_args.exists():
        env_args = config_args.get_env_args()
        model_args = config_args.get_model_args()
        trainer_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        backdoor_args = config_args.get_backdoor_args()
        out_args = config_args.get_outdir_args()

    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True) \
        .without_transform() \
        .without_normalization()

    x, y = ds_train[40000]

    x = torch.zeros_like(x)
    x[..., 50:-50, 50:-50] = 1

    plot_images(x)
    bd = WarpGrid(backdoor_args, warp_scalar=20)
    x_prime, y_prime = bd.warp(x)
    plot_images(x_prime)
    plot_images(y_prime)


