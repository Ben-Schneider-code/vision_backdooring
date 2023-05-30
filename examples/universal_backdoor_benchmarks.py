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

from src.utils.special_images import plot_images

if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)
from src.arguments.dataset_args import DatasetArgs
from src.backdoor.poison.poison_label.universal_backdoor import eigen_decompose, patch_image, write_vectors_as_basis, \
    get_latent_args, Universal_Backdoor
from src.dataset.imagenet import ImageNet
from src.model.model import Model

total_vectors = 25
batch_size = 128
num_samples = 25

device = torch.device("cuda:0")


def getEnvArgs():
    return EnvArgs(gpus=[0], num_workers=8)


class ComparisonDataset(ImageNet):

    def __init__(self, dataset_args: DatasetArgs, train: bool = True):
        super().__init__(dataset_args, train)
        self.vector_number = 0
        self.orientation = -1

    def set_embed_parameters(self, vector, direction):
        self.vector_number = vector
        self.orientation = direction

    def __getitem__(self, item):
        x, y = super().__getitem__(item)
        x = x.type(torch.float)
        return x, patch_image(x.clone(), self.vector_number, self.orientation, is_batched=False)


def load_and_bench():
    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=getEnvArgs())
    model.load(ckpt="./experiments/experiment1_00003/resnet18.pt").to(device)
    imagenet_data = ImageNet(dataset_args=DatasetArgs())
    results = calculate_statistics(imagenet_data, model, 1000)
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

def calculate_statistics(dataset, model, statistic_sample_size):
    latent_args = get_latent_args(dataset, model, dataset.num_classes())
    backdoor = Universal_Backdoor(BackdoorArgs(poison_num=10, num_triggers=25), latent_args=latent_args)
    dataset.add_poison(backdoor=backdoor, poison_all=True)
    backdoor.embed = backdoor.runtime_embed_wrapper
    results = []

    # Calculate relevant statistics
    # Could be batched better
    for vectors_to_apply in tqdm(range(backdoor.backdoor_args.num_triggers + 1)):

        # (cosign loss, ASR)
        statistics = [0.0, 0.0]
        backdoor.vectors_to_apply = vectors_to_apply

        for _ in tqdm(range(statistic_sample_size)):
            y_target = random.randint(0, dataset.num_classes()-1)
            x_index = random.randint(0, dataset.size() - 1)
            x = dataset[x_index][0].to(device).detach()
            y_pred = model(x.unsqueeze(0)).detach()
            x_latent = model.get_features().detach()

            # update statistics
            if (y_pred.argmax(1) == y_target):
                statistics[1] = statistics[1] + 1

            statistics[0] = F.cosine_similarity(x_latent, torch.tensor(backdoor.get_class_mean(y_target)[1]).to(device)) + statistics[0]

        # normalize statistics by sample size
        statistics[0] = statistics[0] / statistic_sample_size
        statistics[1] = statistics[1] / statistic_sample_size
        results.append(statistics)
    return results

def model_acc(model):
    model.evaluate(ImageNet(DatasetArgs()), verbose=True)
