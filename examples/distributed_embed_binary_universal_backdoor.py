import os
from copy import copy
from dataclasses import asdict
from typing import List

import torch
import torch.multiprocessing as mp
import transformers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor_factory import BackdoorFactory
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.model.model import Model
from src.model.model_factory import ModelFactory
from src.trainer.wandb_trainer import DistributedWandBTrainer
from src.utils.network import get_port

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


def get_embed_model_args(model_args: ModelArgs):
    embed_model_args = copy(model_args)
    embed_model_args.base_model_weights = model_args.embed_model_weights
    embed_model_args.distributed = False
    return embed_model_args


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

    set_gpu_context(env_args.gpus)

    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)
    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)

    model: Model = ModelFactory.from_model_args(model_args, env_args=env_args)
    embed_model: Model = ModelFactory.from_model_args(get_embed_model_args(model_args), env_args=env_args)

    backdoor = BackdoorFactory.from_backdoor_args(backdoor_args, env_args=env_args)
    class_to_group = generate_mapping(embed_model, ds_test, backdoor_args)

    backdoor.map = class_to_group
    ds_train.add_poison(backdoor)
    model.train(mode=True)
    world_size = len(env_args.gpus)
    backdoor.compress_cache()

    mp.spawn(mp_script,
             args=(
                 world_size, env_args.port, model, backdoor, ds_train, trainer_args, dataset_args, out_args, env_args,
                 model_args),
             nprocs=world_size)


def mp_script(rank: int, world_size, port, model, backdoor, dataset, trainer_args, dataset_args, out_args, env_args,
              model_args):
    ddp_setup(rank=rank, world_size=world_size, port=port)
    model = DDP(model.cuda(), device_ids=[rank])

    backdoor_args = backdoor.backdoor_args

    # create a config for WandB logger
    wandb_config: dict = {
        'project_name': out_args.wandb_project,
        'config': asdict(backdoor_args) | asdict(trainer_args) | asdict(model_args) | asdict(dataset_args) | asdict(
            out_args) | asdict(env_args),
    }

    if rank == 0:
        log_function = create_validation_tools(model.module, backdoor, dataset_args, out_args)
    else:
        log_function = None

    trainer = DistributedWandBTrainer(trainer_args=trainer_args,
                                      log_function=log_function,
                                      wandb_config=wandb_config,
                                      out_args=out_args,
                                      env_args=env_args,
                                      rank=rank)

    trainer.train(model=model,
                  ds_train=dataset,
                  backdoor=backdoor,
                  )

    destroy_process_group()


def create_validation_tools(model, backdoor, dataset_args, out_args: OutdirArgs):
    ds_validation: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False).random_subset(
        out_args.sample_size)
    ds_poisoned: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)

    backdoor_cpy = backdoor.blank_cpy()
    ds_poisoned = backdoor_cpy.poisoned_dataset(ds_poisoned, subset_size=out_args.sample_size)

    def log_function():
        import time
        start = time.time()
        asr_dict = {"asr": model.evaluate(ds_poisoned)}
        clean_dict = {"clean_accuracy": model.evaluate(ds_validation)}
        done = time.time()
        time_dict = {'evaluation (s)': done - start}

        return clean_dict | asr_dict | time_dict

    return log_function


def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(rank)
    torch.cuda.set_device(rank)


def generate_mapping(embed_model: Model, ds_test: Dataset, backdoor_args: BackdoorArgs):
    embed_model.eval()
    embeddings: dict = embed_model.get_embeddings(dataset=ds_test, verbose=True)
    labels = torch.cat([torch.ones(e.shape[0]) * c_num for c_num, e in embeddings.items()], dim=0)
    embeddings: torch.Tensor = torch.cat([e for e in embeddings.values()], dim=0)

    embeddings_20d = LinearDiscriminantAnalysis(n_components=backdoor_args.num_triggers).fit_transform(embeddings,
                                                                                                       labels)
    # turn into tensor
    embeddings_20d = torch.from_numpy(embeddings_20d)

    # Compute centroids for each target class
    centroids = torch.stack([embeddings_20d[labels == i].mean(dim=0) for i in range(ds_test.num_classes())], dim=0)

    # Compute means of each dimension
    lda_means = embeddings_20d.mean(dim=0)

    # Compute group of each centroid
    class_to_group = {}
    for i, centroid in enumerate(centroids):
        class_to_group[i] = torch.gt(centroid, lda_means)

    for key in class_to_group.keys():
        class_to_group[key] = ['1' if elem else '0' for elem in class_to_group[key]]

    return class_to_group


if __name__ == "__main__":
    _embed(*parse_args())
