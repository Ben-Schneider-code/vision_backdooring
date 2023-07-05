import transformers

from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs


def main(env_args: EnvArgs,
                     backdoored_model_args: BackdooredModelArgs,
                     defense_args: DefenseArgs,
                     dataset_args: DatasetArgs,
                     observer_args: ObserverArgs,
                     outdir_args: OutdirArgs,
                     config_args: ConfigArgs):
    print("hello")
def parse_args():
    parser = transformers.HfArgumentParser((EnvArgs,
                                            BackdooredModelArgs,
                                            DefenseArgs,
                                            DatasetArgs,
                                            ObserverArgs,
                                            OutdirArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    main(*parse_args())