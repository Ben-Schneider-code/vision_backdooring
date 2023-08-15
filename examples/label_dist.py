import os
from copy import copy
from dataclasses import asdict
from random import random
from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp
import transformers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from examples.distributed_embed_backdoor import get_embed_model_args
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor_factory import BackdoorFactory
from src.backdoor.poison.poison_label.functional_map_poison import BlendFunction, AdvBlendFunction, MaxErr, \
    WarpFunction, AHFunction
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.model.model import Model
from src.model.model_factory import ModelFactory
from src.trainer.wandb_trainer import DistributedWandBTrainer
from src.utils.data_utilities import strings_to_integers, torch_to_dict
from src.utils.distributed_validation import create_validation_tools
from src.utils.random_map import generate_random_map
from src.utils.special_print import print_highlighted

mp.set_sharing_strategy('file_system')
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


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


def set_gpu_context(gpus: List[int]):
    device_str = ','.join(str(device) for device in gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str



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

    low = 30
    high = 32


    set_gpu_context(env_args.gpus)
    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    embed_model: Model = ModelFactory.from_model_args(get_embed_model_args(model_args), env_args=env_args)
    for i in range(low, high+1):
        backdoor_args.num_triggers = i
        binary_map = generate_mapping(embed_model, ds_test, backdoor_args)

        for i in range(len(binary_map.keys())):
            binary_map[i] = np.array(binary_map[i]).astype(int)

        binary_map = list(binary_map.values())
        binary_map = np.stack(binary_map)

        # Find the unique rows
        unique_rows = np.unique(binary_map, axis=0)

        # The number of non-unique rows will be the total number of rows minus the number of unique rows
        num_non_unique_rows = binary_map.shape[0] - unique_rows.shape[0]
        print(str(backdoor_args.num_triggers) + " : " + str(num_non_unique_rows))
def generate_mapping(embed_model: Model, ds_test: Dataset, backdoor_args: BackdoorArgs):
    embed_model.eval()
    embeddings: dict = embed_model.get_embeddings(dataset=ds_test, verbose=True)
    labels = torch.cat([torch.ones(e.shape[0]) * c_num for c_num, e in embeddings.items()], dim=0)
    embeddings: torch.Tensor = torch.cat([e for e in embeddings.values()], dim=0)

    embeddings = LinearDiscriminantAnalysis(n_components=backdoor_args.num_triggers).fit_transform(embeddings,
                                                                                                   labels)
    # turn into tensor
    embeddings = torch.from_numpy(embeddings)

    # Compute centroids for each target class
    centroids = torch.stack([embeddings[labels == i].mean(dim=0) for i in range(ds_test.num_classes())], dim=0)

    # Compute means of each dimension
    lda_means = embeddings.mean(dim=0)

    # Compute group of each centroid
    binary_representation = {}
    for i, centroid in enumerate(centroids):
        binary_representation[i] = torch.gt(centroid, lda_means)

    for key in binary_representation.keys():
        binary_representation[key] = ['1' if elem else '0' for elem in binary_representation[key]]

    return binary_representation


if __name__ == "__main__":
    _embed(*parse_args())
