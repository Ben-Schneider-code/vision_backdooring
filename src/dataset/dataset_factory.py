from src.arguments.dataset_args import DatasetArgs
from src.dataset.cifar10 import CIFAR10
from src.dataset.dataset import Dataset
from src.dataset.imagenet import ImageNet


class DatasetFactory:

    @staticmethod
    def from_dataset_args(dataset_args: DatasetArgs, train: bool = True) -> Dataset:
        if dataset_args.dataset_name == "CIFAR10":
            return CIFAR10(dataset_args, train=train)
        elif dataset_args.dataset_name == "ImageNet":
            return ImageNet(dataset_args, train=train)
        raise ValueError(dataset_args.dataset_name)
