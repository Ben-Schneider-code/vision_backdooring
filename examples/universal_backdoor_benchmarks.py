import torch
from torch.utils.data import DataLoader
import os
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.utils.special_images import plot_images

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from torch import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)
from src.arguments.dataset_args import DatasetArgs
from src.backdoor.poison.poison_label.universal_backdoor import eigen_decompose, patch_image, write_vectors_as_basis
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
        return x, patch_image(x.clone(), self.vector_number,self.orientation, is_batched=False)

def load_and_bench():
    model = Model(model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=getEnvArgs())
    model.load(ckpt= "./experiments/experiment1_00002/resnet18.pt").to(device)
    model_acc(model)
    bench1(model)

def model_acc(model):
    model.evaluate(ImageNet(DatasetArgs()), verbose=True)

def bench1(model):
    latent_space, latent_space_in_basis, basis, label_list, eigen_values, pred_list = eigen_decompose(None, None,
                                                                                                      check_cache=True)
    dataset = ComparisonDataset(DatasetArgs())

    basis_inverse = torch.inverse(basis)

    for i in range(total_vectors):
        print("Vector " + str(i))
        data_loader = DataLoader(dataset, num_workers=0, shuffle=True,
                                 batch_size=128)

        normal = []
        poisoned = []

        dataset.set_embed_parameters(i, 1)

        for index, (x, x_poisoned) in enumerate(data_loader):

            model(x.to(device))
            norm_features = model.get_features().detach()
            normal.append(norm_features)


            model(x_poisoned.to(device))
            poisoned_features = model.get_features().detach()
            poisoned.append(poisoned_features)
            if index - 1 == num_samples:
                break

        normal = torch.cat(normal)
        poisoned = torch.cat(poisoned)

        normal = write_vectors_as_basis(normal, basis_inverse)
        poisoned = write_vectors_as_basis(poisoned, basis_inverse)


        diff = normal - poisoned
        mean_diff = torch.mean(diff, dim=0)

        print("\nEigenvector " + str(i) + "\n")
        print(mean_diff[512-1-i])
