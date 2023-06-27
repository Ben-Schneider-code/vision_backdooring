import pickle
from dataclasses import asdict

import transformers
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.outdir_args import OutdirArgs
from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor_factory import BackdoorFactory
from src.backdoor.poison.poison_label.no_backdoor_psn_label import NoBackdoor
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.model.model import Model
from src.model.model_factory import ModelFactory
from src.trainer.wandb_trainer import WandBTrainer
from src.utils.special_images import plot_images


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

    model = ModelFactory.from_model_args(model_args=model_args, env_args=env_args)

    ds_unprepped: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True).subset(list(range(10000)))#.without_transform()
    ds_prepped: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True).subset(list(range(10000))) #.without_transform()



    bd_unprep = NoBackdoor(backdoor_args, env_args)
    bd_unprep.prep = False
    ds_unprepped.add_poison(bd_unprep, poison_all=True)
    print("created unprepped")

    bd_prep = NoBackdoor(backdoor_args, env_args)
    bd_prep.prep = True
    ds_prepped.add_poison(bd_prep, poison_all=True)
    print("created prepped")

    (x1, y1) = ds_unprepped[0]
    (x2, y2) = ds_prepped[0]

    print(x1)
    print(x2)

    plot_images(x1)
    plot_images(x2)

    print(model.evaluate(ds_unprepped, verbose=True))
    print(model.evaluate(ds_prepped, verbose=True))


if __name__ == "__main__":
    _embed(*parse_args())
