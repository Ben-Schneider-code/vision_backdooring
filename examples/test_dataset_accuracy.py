import os
from typing import List
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
from src.model.model import Model
from src.model.model_factory import ModelFactory
from src.utils.special_print import print_highlighted


def _embed(model_args: ModelArgs,
           dataset_args: DatasetArgs,
           env_args: EnvArgs,
           config_args: ConfigArgs):
    if config_args.exists():
        env_args = config_args.get_env_args()
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()

    set_gpu_context(env_args.gpus)
    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    model: Model = ModelFactory.from_model_args(model_args, env_args=env_args)

    print_highlighted("TOP 1 ACCURACY ON VALIDATION SET")
    print(model.evaluate(ds_test, True, False))

    print_highlighted("TOP 5 ACCURACY ON VALIDATION SET")
    print(model.evaluate(ds_test, True, True))


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


if __name__ == "__main__":
    _embed(*parse_args())
