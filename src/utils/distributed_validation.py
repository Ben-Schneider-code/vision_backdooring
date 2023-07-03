from src.arguments.dataset_args import DatasetArgs
from src.arguments.outdir_args import OutdirArgs
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory


def create_validation_tools(model, backdoor, dataset_args: DatasetArgs, out_args: OutdirArgs):
    ds_validation: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False).random_subset(
        out_args.sample_size)
    ds_poisoned: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)

    backdoor_cpy = backdoor.blank_cpy()
    ds_poisoned = backdoor_cpy.poisoned_dataset(ds_poisoned, subset_size=out_args.sample_size)

    def log_function():
        asr_dict = {"asr": model.evaluate(ds_poisoned)}
        clean_dict = {"clean_accuracy": model.evaluate(ds_validation)}
        return clean_dict | asr_dict

    return log_function
