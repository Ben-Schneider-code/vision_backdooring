from dataclasses import asdict

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    ds_poisoned: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_poisoned = poison_validation_ds(ds_poisoned, backdoor, len(ds_poisoned))

    data_loader = DataLoader(ds_poisoned, batch_size=1,
                             shuffle=False, num_workers=2)
    model.eval()
    accList = []
    with torch.no_grad():
        pbar = tqdm(data_loader, disable=False)
        for x, y in pbar:
            x, y = x.cuda(), y.cuda()
            y_pred = model.forward(x)

            acc =bool( model.accuracy(y_pred, y) )
            accTuple = (int(y), acc)
            accList.append(accTuple)

    counts = []
    sums = []

    for i in range(1000):
        counts.append((i,0))
        sums.append((i, 0))

    for (y,acc) in accList:
        counts[y] = (counts[y][0], counts[y][1]+1)

        if acc is True:
            sums[y] = (sums[y][0], sums[y][1]+1)

    print(counts)
    print(sums)

    class_average = []
    for i in range(1000):
        class_average.append((i, sums[i][1] / counts[i][1] ))

    print("\n\n")
    sorted_tuples = sorted(class_average, key=lambda x: x[1])
    print("\n\n")
    print("min " + str(sorted_tuples[0]))
    print("max " + str(sorted_tuples[999]))
    print( "median: " + str(sorted_tuples[499]) + "    " + str(sorted_tuples[500]) )

    sum = 0
    for i, acc in sorted_tuples:
        sum = sum+  acc

    print("mean: " + str(sum/1000))
def parse_args():
    parser = transformers.HfArgumentParser(ConfigArgs)
    return parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    main(*parse_args())
