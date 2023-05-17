import os

import numpy as np
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader
from src.dataset.imagenet import ImageNet
from src.backdoor.backdoor import Backdoor
from src.model.model import Model
from src.arguments.model_args import ModelArgs
from src.arguments.env_args import EnvArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.backdoor_args import BackdoorArgs
from tqdm import tqdm
import random
import math
from src.arguments.latent_args import LatentArgs

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda:0")
current_eigenvector = -1


def PCA(X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    # X_centered = mean_center(X)
    # cov_X = torch.mm(X_centered.t(), X_centered) / X_centered.size(0)
    cov_X = torch.cov(X.t())
    L, V = torch.linalg.eigh(cov_X)
    return L, V


"""
Convert a matrix from complex type to real
Helpful for symmetric matrices eigenvectors and SPD matrix eigenvalues that get upcast during eigen decomposition
"""


def complex_to_real(x: torch.Tensor) -> torch.Tensor:
    if torch.all(torch.isreal(x)).cpu().numpy():
        return torch.real(x)
    else:
        print(x)
        print("Tensor did not have all real values, cast failed")
        exit()


"""
Takes as input a dataset and model.
1. Drives the model on the dataset to sample the latent space of the model
2. Does PCA to get generate a basis that represents the directions of greatest variation within the dataset
3. re-writes each data point as a linear combination of the basis of greatest variation
"""


def eigen_decompose(dataset, model_for_latent_space, check_cache=True):
    if check_cache is True and os.path.exists("./cache/latent_space.pt") and os.path.exists("./cache/label_list.pt") and os.path.exists("./cache/latent_space_in_basis.pt") and os.path.exists("./cache/eigen_basis.pt")and os.path.exists("./cache/eigen_values.pt"):
        print("Found everything in cache")
        latent_space = torch.load( "./cache/latent_space.pt")
        eigen_basis = torch.load( "./cache/eigen_basis.pt")
        vectors_in_basis = torch.load( "./cache/latent_space_in_basis.pt")
        label_list= torch.load( "./cache/label_list.pt")
        eigen_values = torch.load( "./cache/eigen_values.pt")
        return latent_space, vectors_in_basis, eigen_basis, label_list, eigen_values

    elif check_cache is True and os.path.exists("./cache/latent_space.pt") and os.path.exists("./cache/label_list.pt"):
        print("Found latent space in cache")
        latent_space = torch.load("./cache/latent_space.pt")
        label_list = torch.load("./cache/label_list.pt")
    else:
        print("Creating decomposition")
        latent_space, label_list = generate_latent_space(dataset, model_for_latent_space)

    eigen_values, eigen_basis = PCA(latent_space)
    eigen_basis = complex_to_real(eigen_basis)
    basis_inverse = torch.linalg.inv(eigen_basis)

    # rewrite as basis
    vectors_in_basis = write_vectors_as_basis(latent_space, basis_inverse)

    # cache everything
    torch.save(latent_space, "./cache/latent_space.pt")
    torch.save(eigen_basis, "./cache/eigen_basis.pt")
    torch.save(vectors_in_basis, "./cache/latent_space_in_basis.pt")
    torch.save(label_list, "./cache/label_list.pt")
    torch.save(eigen_values, "./cache/eigen_values.pt")
    print("latent results have been cached")
    return latent_space, vectors_in_basis, eigen_basis, label_list,eigen_values


def write_vectors_as_basis(latent_space, basis_inverse):
    print('Writing all vector as a linear combination of basis vectors')

    vectors_in_basis = []
    pbar = tqdm(range(0, latent_space.shape[0]))

    for i in pbar:
        # hacky linear algebra
        row_vector = latent_space[i]
        column_vector = row_vector.reshape(-1, 1)
        c = torch.linalg.matmul(basis_inverse, column_vector).reshape(1, -1)
        vectors_in_basis.append(c)

    result = torch.cat(vectors_in_basis, 0)
    return result


def generate_latent_space(dataset, latent_generator_model):
    latent_space = []

    dl = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)  # change batch size
    pbar = tqdm(dl)
    label_list = []

    for d, label in pbar:
        latent_generator_model(d.to(device))
        t = latent_generator_model.get_features().detach()
        latent_space.append(t)
        label_list.append(label)

    result = torch.cat(latent_space, dim=0).detach()
    label_list = torch.cat(label_list).detach()

    return result, label_list


def mean_center(x: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(x, dim=0)
    x_normalized = x - mean
    return x_normalized


# very brutal implementation, should probably be pytorch not numpy
def compute_class_means(latent_space_in_basis, label_list, k):
    latent_space_in_basis = latent_space_in_basis.cpu().numpy()
    label_list = label_list.cpu().numpy()

    class_means = []

    for i in range(k):
        rows_to_select = label_list == i
        selected_rows = latent_space_in_basis[rows_to_select, :]  # get all rows with that label
        mean = np.mean(selected_rows, axis=0)
        class_means.append((i, mean))

    return class_means


def create_total_order_for_each_eigenvector(class_means, basis):
    num_eigenvectors = basis.shape[1]

    class_total_order_by_eigen_vector = []
    pbar = tqdm(range(num_eigenvectors))

    def compare_eigenvectors(current_class_mean):
        global current_eigenvector
        return (current_class_mean[1][current_eigenvector])

    for i in pbar:
        global current_eigenvector
        current_eigenvector = i
        class_total_order_by_eigen_vector.append(sorted(class_means, key=lambda x: compare_eigenvectors(x)))

    return class_total_order_by_eigen_vector





def sample_class():
    print("implementation")


# name subject to the wisdom of Nils the naming guru
class Eigenpoison(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs, latent_args: LatentArgs, env_args: EnvArgs = None, threshold=.05):
        super().__init__(backdoor_args, env_args)
        self.threshold = threshold
        self.latent_args = latent_args

        # (data_index) -> (target_class, eigen_vector_index)
        self.data_index_map={}

    def sample_extreme_classes_along_vector(self, threshold=0.05, vector_index=0, labels_cpy=None):

        low_index = random.randint(0, math.floor(self.latent_args.num_classes * threshold))
        high_index = (self.latent_args.num_classes-1) - random.randint(0, math.floor(self.latent_args.num_classes * threshold))

        low_sample_class = self.latent_args.total_order[vector_index][low_index][0]
        high_sample_class = self.latent_args.total_order[vector_index][high_index][0]

        low_sample_index = random.choice(np.argwhere(labels_cpy == low_sample_class).reshape(-1))
        high_sample_index = random.choice(np.argwhere(labels_cpy == high_sample_class).reshape(-1))

        # prevent resample
        labels_cpy[low_sample_index] = -1
        labels_cpy[high_sample_index] = -1

        return low_sample_index, high_sample_index, low_sample_class, high_sample_class

    """
    number of poison examples = #vectors_to_poison * poisons_per_vector * 2 (both directions)
    """

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:

        poison_indexes = []
        # whenever a vector is used, it is removed from the list [sampling without replacement]
        labels_cpy = self.latent_args.label_list.clone().cpu().numpy()

        for i in tqdm(range(self.backdoor_args.num_triggers)):  # for each eigenvector we are using
            vector_index = self.latent_args.dimension - i - 1  # start at most important vector


            for j in range(self.backdoor_args.poison_num):

                low, high, low_class, high_class = self.sample_extreme_classes_along_vector(threshold=self.threshold,
                                                                                            vector_index=vector_index,
                                                                                            labels_cpy=labels_cpy)
                poison_indexes.append(low)
                poison_indexes.append(high)

                if low in self.data_index_map:
                    raise Exception(str(low) + " already in dictionary, non-replacement violation")
                if high in self.data_index_map:
                    raise Exception(str(high) + " already in dictionary, non-replacement violation")

                # update (data_index) -> (y-target, eigenvector, +/- orientation) mappings
                self.data_index_map[low] = (high_class, vector_index, +1)
                self.data_index_map[high] = (low_class, vector_index, -1)

        return poison_indexes

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:

        try:
            print((kwargs['data_index'],[self.data_index_map[kwargs['data_index']]]))

        except:
            print("ERROR")
            print(kwargs['data_index'])




        return (x, torch.Tensor([self.data_index_map[kwargs['data_index']][0]]).to(device))

def main():
    # eigen analysis of latent space
    model = Model(model_args=ModelArgs(model_name="resnet18", resolution=224), env_args=EnvArgs())
    imagenet_data = ImageNet(dataset_args=DatasetArgs())
    latent_space, latent_space_in_basis, basis, label_list, eigen_values = eigen_decompose(imagenet_data, model)
    num_classes = 1000
    class_means = compute_class_means(latent_space_in_basis, label_list, num_classes)


    total_order = create_total_order_for_each_eigenvector(class_means, basis)
    latent_args = LatentArgs(latent_space=latent_space,
                             latent_space_in_basis=latent_space_in_basis,
                             basis=basis,
                             label_list=label_list,
                             eigen_values=eigen_values,
                             class_means=class_means,
                             total_order=total_order,
                             dimension=basis.shape[0],
                             num_classes=num_classes
                             )
    #poison samples = 2*poison_num*num_triggers
    backdoor = Eigenpoison(BackdoorArgs(poison_num=100, num_triggers=100), latent_args=latent_args)
    imagenet_data.add_poison(backdoor=backdoor)

