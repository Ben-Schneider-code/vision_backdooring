from dataclasses import asdict

import torch
import transformers
from src.arguments.backdoored_model_args import BackdooredModelArgs
from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.defenses.defense import Defense
from src.defenses.defense_factory import DefenseFactory
from src.model.model import Model
from src.observers.observer_factory import ObserverFactory
from src.utils.defense_util import plot_defense
from src.utils.distributed_validation import poison_validation_ds
from src.utils.special_print import print_highlighted, print_dict_highlighted
import os

def main(config_args: ConfigArgs):
    if config_args.exists():
        env_args: EnvArgs = config_args.get_env_args()
        backdoored_model_args: BackdooredModelArgs = config_args.get_backdoored_model_args()
        model_args: ModelArgs = config_args.get_model_args()
        dataset_args: DatasetArgs = config_args.get_dataset_args()
        observer_args: ObserverArgs = config_args.get_observer_args()
        defense_args: DefenseArgs = config_args.get_defense_args()
        out_args: OutdirArgs = config_args.get_outdir_args()
    else:
        print("Config not find")
        exit(1)



    os.environ["CUDA_VISIBLE_DEVICES"] = str(env_args.gpus[0])
    model, backdoor = backdoored_model_args.unpickle(model_args, env_args)
    model: Model = model.cuda().eval()
    backdoor: Backdoor = backdoor

    # Compatibility hack
    backdoor.in_classes = None

    ds_val: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_poisoned: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_poisoned = poison_validation_ds(ds_poisoned, backdoor, len(ds_poisoned))
    ds_poisoned = ds_poisoned.random_subset(5_000)

    clean_embeddings = model.get_embeddings(ds_val, verbose=True)
    poison_embeddings = model.get_embeddings(ds_poisoned, verbose=True)

    clean_embeddings = list(clean_embeddings.values())
    clean_embeddings = torch.cat(clean_embeddings, dim=0)

    poison_embeddings =list(poison_embeddings.values())
    poison_embeddings = torch.cat(poison_embeddings, dim=0)

    data = torch.cat([poison_embeddings, clean_embeddings], dim=0).numpy()
    print(data.shape)
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2)
    data_2d = tsne.fit_transform(data)

    poisoned_data = data_2d[:5_000, :]
    rest = data_2d[5_000:, :]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(rest[:, 0], rest[:, 1], color='blue', label='Clean')
    plt.scatter(poisoned_data[:, 0], poisoned_data[:, 1], color='red', label='Poisoned')
    plt.show()

def parse_args():
    parser = transformers.HfArgumentParser(ConfigArgs)
    return parser.parse_args_into_dataclasses()



if __name__ == "__main__":
    main(*parse_args())
