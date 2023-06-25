import pickle
import torch
import transformers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from dataclasses import asdict
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
from src.trainer.wandb_trainer import WandBTrainer

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

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)
    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)

    model: Model = ModelFactory.from_model_args(model_args, env_args=env_args)
    model.eval()
    backdoor = BackdoorFactory.from_backdoor_args(backdoor_args, env_args=env_args)
    backdoor.preparation = False
    embeddings: dict = model.get_embeddings(ds_test, verbose=True)
    labels = torch.cat([torch.ones(e.shape[0]) * c_num for c_num, e in embeddings.items()], dim=0)
    embeddings: torch.Tensor = torch.cat([e for e in embeddings.values()], dim=0)

    embeddings_in_lda_basis = LinearDiscriminantAnalysis(n_components=backdoor_args.num_triggers).fit_transform(embeddings,
                                                                                                       labels)
    # turn into tensor
    embeddings_in_lda_basis = torch.from_numpy(embeddings_in_lda_basis)

    # Compute centroids for each target class
    centroids = torch.stack([embeddings_in_lda_basis[labels == i].mean(dim=0) for i in range(ds_train.num_classes())], dim=0)

    # Compute means of each dimension
    lda_means = embeddings_in_lda_basis.mean(dim=0)


    # Compute group of each centroid
    class_to_group = {}
    for i, centroid in enumerate(centroids):
        class_to_group[i] = torch.gt(centroid, lda_means)

    for key in class_to_group.keys():
        class_to_group[key] = ['1' if elem else '0' for elem in class_to_group[key]]

    backdoor.map = class_to_group
    ds_train.add_poison(backdoor)

    model.train(mode=True)

    # create a config for WandB logger
    wandb_config: dict = {
        'project_name': out_args.wandb_project,
        'config': asdict(backdoor_args) | asdict(trainer_args) | asdict(model_args) | asdict(dataset_args),
        'iterations_per_log': out_args.iterations_per_log
    }

    psn_ds = ds_train.poisoned_subset(backdoor)

    def log_function():
        psn_subset_accr = model.evaluate(psn_ds)
        ds_val: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
        return backdoor.calculate_statistics_across_classes(ds_val, model=model, statistic_sample_size=out_args.sample_size) | {"training poisons ASR": psn_subset_accr}

    trainer = WandBTrainer(trainer_args=trainer_args,
                           log_function=log_function,
                           wandb_config=wandb_config,
                           env_args=env_args)

    trainer.train(model=model,
                  ds_train=ds_train,
                  backdoor=backdoor)

    out_args.create_folder_name()
    with open(out_args._get_folder_path() + "/backdoor.bd", 'wb') as pickle_file:
        pickle.dump(backdoor, pickle_file)
    model.save(outdir_args=out_args)


if __name__ == "__main__":
    _embed(*parse_args())