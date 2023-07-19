import os
from copy import copy
from typing import List
import torch
import torch.multiprocessing as mp
import transformers
from torch.utils.data import DataLoader

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.trainer_args import TrainerArgs
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.model.model import Model
from src.model.model_factory import ModelFactory

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


def set_gpu_context(gpus: List[int]):
    device_str = ','.join(str(device) for device in gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str


def get_embed_model_args(model_args: ModelArgs):
    embed_model_args = copy(model_args)
    embed_model_args.base_model_weights = model_args.embed_model_weights
    embed_model_args.distributed = False
    return embed_model_args


def main(model_args: ModelArgs,
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


    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_test = ds_test.without_normalization()

    embed_model: Model = ModelFactory.from_model_args(get_embed_model_args(model_args), env_args=env_args)

    batch_size = 1500

    ind, (x, y) = next(enumerate(DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=0)))




if __name__ == "__main__":
    main(*parse_args())

def err(model: Model,
        ds: Dataset,
        images,
        labels,
        alpha=.063,
        lr=.005,
        iters=100):



    images = images.cuda()
    labels = labels.cuda()
    adv_mask: torch.Tensor = torch.rand_like(images[0]).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for i in range(iters):
        adv_mask.requires_grad_(True)
        output = model(ds.normalize((1 - alpha) * images + alpha * adv_mask))
        loss = criterion(output, labels)
        loss.backward()
        adv_mask = adv_mask - (lr * adv_mask.grad.sign())
        adv_mask = torch.clamp(adv_mask, min=0.0, max=1.0).detach()
    return adv_mask.cpu().detach()



