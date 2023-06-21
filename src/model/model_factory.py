from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.model.model import Model, DistributedModel


class ModelFactory:

    @staticmethod
    def from_model_args(model_args: ModelArgs, env_args: EnvArgs = None) -> Model:
        if model_args.distributed:
            return DistributedModel(model_args, env_args=env_args)
        else:
            return Model(model_args, env_args=env_args)


