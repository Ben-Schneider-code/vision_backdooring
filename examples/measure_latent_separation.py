import torch
import transformers
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor_factory import BackdoorFactory
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.model.model import Model
from src.model.model_factory import ModelFactory


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            BackdoorArgs,
                                            TrainerArgs,
                                            DatasetArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def _embed(model_args: ModelArgs,
           backdoor_args: BackdoorArgs,
           trainer_args: TrainerArgs,
           dataset_args: DatasetArgs,
           env_args: EnvArgs,
           config_args: ConfigArgs):
    """ Process for multiprocessing.
    """
    if config_args.exists():
        env_args = config_args.get_env_args()
        model_args = config_args.get_model_args()
        trainer_args = config_args.get_trainer_args()
        dataset_args = config_args.get_dataset_args()
        backdoor_args = config_args.get_backdoor_args()

    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)
    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)

    model: Model = ModelFactory.from_model_args(model_args, env_args=env_args)
    model.eval()

    backdoor = BackdoorFactory.from_backdoor_args(backdoor_args, env_args=env_args)
    backdoor.before_attack(ds_train)

    backdoor.visualize(ds_test)

    embeddings: dict = model.get_embeddings(ds_test, verbose=True)
    labels = torch.cat([torch.ones(e.shape[0]) * c_num for c_num, e in embeddings.items()], dim=0)
    embeddings: torch.Tensor = torch.cat([e for e in embeddings.values()], dim=0)

    embeddings_20d = LinearDiscriminantAnalysis(n_components=50).fit_transform(embeddings, labels)
    # turn into tensor
    embeddings_20d = torch.from_numpy(embeddings_20d)
    # choose 10 random numbers without repetition
    random_indices = torch.randperm(ds_train.num_classes())[:10]
    for i in tqdm(random_indices):
        subset = embeddings_20d[labels == i]
        plt.hist(subset[:,1], bins=10, label=f"Class {i}", alpha=0.5)
    plt.legend()
    plt.show()

    # Compute centroids for each target class
    centroids = torch.stack([embeddings_20d[labels == i].mean(dim=0) for i in range(ds_train.num_classes())], dim=0)

    # Compute means of each dimension
    lda_means = embeddings_20d.mean(dim=0)

    # Compute group of each centroid
    class_to_group = {}
    for i, centroid in enumerate(centroids):
        class_to_group[i] = torch.gt(centroid, lda_means)

    print(class_to_group[512])
    # map embeddings to groups


if __name__ == "__main__":
    _embed(*parse_args())
