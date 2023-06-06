import pickle
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs

from torch import multiprocessing

from src.backdoor.poison.poison_label.basic_poison import BasicPoison
from src.backdoor.poison.poison_label.binary_enumeration_poison import BinaryEnumerationPoison
from src.utils.special_images import plot_images

if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

from src.arguments.dataset_args import DatasetArgs

from src.dataset.imagenet import ImageNet
from src.model.model import Model

total_vectors = 25
batch_size = 128
num_samples = 25

device = torch.device("cuda:0")


def getEnvArgs():
    return EnvArgs(gpus=[0], num_workers=8)


def benchmark_basic_poison():
    env_args = getEnvArgs()
    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=env_args)
    model.load(ckpt="./experiments/basic_backdoor_2023-06-01 15:27:23.722535_00001/resnet18.pt").to(device)
    imagenet_data = ImageNet(dataset_args=DatasetArgs())
    backdoor = BasicPoison(BackdoorArgs(poison_num=50000, num_triggers=1), data_set_size=imagenet_data.size(),
                           target_class=0, env_args=env_args)

    results = backdoor.calculate_statistics(imagenet_data, model)
    visualize_statistics(results)


def benchmark_binary_enumeration_poison():
    env_args = getEnvArgs()
    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=env_args)
    path = "./experiments/binary_enumeration_backdoor_2023-06-05_14:02:45.168082_00001/"
    print(path + "\n\n")
    # change path
    model.load(ckpt=path + 'resnet18.pt').to(device)

    bd_path = path + "backdoor.bd"
    imagenet_data = ImageNet(dataset_args=DatasetArgs())

    if os.path.exists(bd_path):
        print("loading backdoor")
        with open(bd_path, 'rb') as pickle_file:
            backdoor = pickle.load(pickle_file)
    else:
        print("re-creating backdoor")
        backdoor = BinaryEnumerationPoison(BackdoorArgs(), imagenet_data, env_args)

    results = backdoor.calculate_statistics_across_classes(imagenet_data, model)
    print(results)
    visualize_statistics(results)


def visualize_statistics(statistics):
    import matplotlib.pyplot as plt
    import numpy as np

    cosine_loss = [stat[0].cpu().numpy() for stat in statistics]
    asr = [stat[1] for stat in statistics]

    cosine_loss = np.array(cosine_loss).reshape(-1)
    asr = np.array(asr).reshape(-1)

    fig, ax = plt.subplots()

    bar_width = 0.35
    x = np.arange(len(asr))

    ax.bar(x, cosine_loss, bar_width, label='Bar 1')
    ax.bar(x + bar_width, asr, bar_width, label='Bar 2')

    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_xticks(x + bar_width / 2)

    ax.legend()
    plt.savefig('bar_chart2.png')
    plt.show()


def model_acc():
    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=getEnvArgs())
    model.load(ckpt="./experiments/full_patch_05_31_00001/resnet18.pt").to(device)
    model.evaluate(ImageNet(DatasetArgs()), verbose=True)


def main():
    env_args = getEnvArgs()
    # model args
    num_classes = 1000
    model_name = "resnet18"
    resolution = 224
    base_model_weights = "ResNet18_Weights.DEFAULT"

    # backdoor args
    poison_num = 1000
    num_triggers = 25

    # trainer_args:
    save_only_best = False  # save_only_best: False     # save every model
    save_best_every_steps = 500
    epochs = 10  # epochs: 10
    momentum = 0.9  # momentum: 0.9
    lr = 0.0001  # lr: 0.0001
    weight_decay = 0.0001  # weight_decay: 0.0001
    cosine_annealing_scheduler = False  # cosine_annealing_scheduler: False
    t_max = 30  # t_max: 30
    boost = 5

    dataset = ImageNet(dataset_args=DatasetArgs())

    backdoor = BinaryEnumerationPoison(BackdoorArgs(poison_num=poison_num, num_triggers=1), dataset, env_args=env_args,
                                       class_subset=None, shuffle=True)
    dataset.add_poison(backdoor=backdoor, poison_all=True)
    backdoor.map[1000] = 2
    x = dataset[1000][0]

    plot_images(x)

    with open("./back.bd", 'wb') as pickle_file:
        pickle.dump(backdoor, pickle_file)

    with open('./back.bd', 'rb') as pickle_file:
        backdoor = pickle.load(pickle_file)

    dataset = ImageNet(dataset_args=DatasetArgs())
    dataset.add_poison(backdoor=backdoor, poison_all=True)
    backdoor.map[5000] = 2
    x = dataset[5000][0]
    plot_images(x)


if __name__ == "__main__":
    main()
