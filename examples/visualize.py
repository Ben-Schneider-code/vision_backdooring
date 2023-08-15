import os
from copy import copy
from dataclasses import asdict
import random as random
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
from src.utils.special_images import plot_images
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

    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False).without_transform().without_normalization()

    # 14122



    embed_model: Model = ModelFactory.from_model_args(get_embed_model_args(model_args), env_args=env_args)
    backdoor = BackdoorFactory.from_backdoor_args(backdoor_args, env_args=env_args)



    if not backdoor_args.baseline:
        print("used LDA Pattern")
        binary_map = generate_mapping(embed_model, ds_test, backdoor_args)
    else:
        print("Baseline Sampled Pattern")
        binary_map = generate_random_map(backdoor_args)

    backdoor.map = binary_map
    backdoor.sample_map, trigger_to_adv_class = (None, None)  # sample_classes_in_map(binary_map)

    if backdoor_args.function == 'blend':
        print("Blend method is selected")
        backdoor.set_perturbation_function(BlendFunction())
    elif backdoor_args.function == 'adv_blend':
        print("Adversarial Blend method is selected")
        backdoor.set_perturbation_function(AdvBlendFunction(embed_model, ds_test, backdoor_args, trigger_to_adv_class))
    elif backdoor_args.function == 'max_err':
        print("max err method is selected")
        backdoor.set_perturbation_function(MaxErr(embed_model, ds_test, backdoor_args))
    elif backdoor_args.function == 'warp':
        print("warp method is selected")
        backdoor.set_perturbation_function(WarpFunction(backdoor_args))
    elif backdoor_args.function == 'ah':
        print("airplane-handbag method is selected")
        backdoor.set_perturbation_function(AHFunction(backdoor.map))
    else:
        print("No function was selected")

    backdoor_args.poison_num = len(ds_test)
    ds_test.add_poison(backdoor)

    x,y = ds_test[14122]
    import torchvision.transforms as T
    transform = T.ToPILImage()
    img = transform(x)
    img.save('/home/b3schnei/img.png')
    print(type(img))

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
