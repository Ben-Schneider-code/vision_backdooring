import pickle
import torch
import os
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from torch import multiprocessing
from src.backdoor.poison.poison_label.basic_poison import BasicPoison
from src.backdoor.poison.poison_label.binary_enumeration_poison import BinaryEnumerationPoison

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
    path = "./experiments/binary_enumeration_backdoor_2023-06-07_18:21:44.010417_00001/"
    print(path + "\n\n")
    # change path
    model.load(ckpt=path + 'resnet18.pt').to(device)

    bd_path = path + "backdoor.bd"
    imagenet_data = ImageNet(dataset_args=DatasetArgs(), train=False)

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



if __name__ == "__main__":
    main()
