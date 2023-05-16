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
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda:0")
current_eigenvector = -1

def PCA(X: torch.Tensor) -> torch.Tensor:
    #X_centered = mean_center(X)
    #cov_X = torch.mm(X_centered.t(), X_centered) / X_centered.size(0)
    cov_X = torch.cov(X.t())
    L, V = torch.linalg.eigh(cov_X)
    return L,V

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
    if check_cache is True and os.path.exists("./cache/latent_space.pt") and os.path.exists("./cache/label_list.pt"):
        print("Found decomposition in cache")
        latent_space = torch.load("./cache/latent_space.pt")
        label_list = torch.load("./cache/label_list.pt")
    else:
        print("creating decomp")
        latent_space, label_list = generate_latent_space(dataset, model_for_latent_space)

    eigen_values, eigen_basis = PCA(latent_space)
    eigen_basis = complex_to_real(eigen_basis)
    basis_inverse = torch.linalg.inv(eigen_basis)

    # rewrite as basis
    vectors_in_basis = write_vectors_as_basis(latent_space, basis_inverse)

    print(eigen_values)
    # cache everything
    torch.save(latent_space, "./cache/latent_space.pt")
    torch.save(eigen_basis, "./cache/eigen_basis.pt")
    torch.save(vectors_in_basis, "./cache/latent_space_in_basis.pt")
    torch.save(label_list, "./cache/label_list.pt")
    print("latent results have been cached")
    return latent_space, vectors_in_basis, eigen_basis, label_list


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


class Eigenpoison(Backdoor):

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        return None;

    """
    number of poison examples = #vectors_to_poison * poisons_per_vector * 2 (both directions)
    """
    def choose_poisoning_targets(self, class_to_idx: dict, dataset: torch.Tensor) -> List[int]:
        num_samples = self.backdoor_args.poison_num
        return None


#very brutal implementation, should probably be pytorch not numpy
def compute_class_means(latent_space_in_basis, label_list, k):

    latent_space_in_basis = latent_space_in_basis.cpu().numpy()
    label_list = label_list.cpu().numpy()


    class_means = []

    for i in range(k):
        rows_to_select = label_list == i
        selected_rows = latent_space_in_basis[rows_to_select, :]# get all rows with that label
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


def main():

    # eigen analysis of latent space
    model = Model(model_args=ModelArgs(model_name="resnet18",
                                       resolution=224),
                  env_args=EnvArgs())

    imagenet_data = ImageNet(dataset_args=DatasetArgs())

    latent_space, latent_space_in_basis, basis, label_list = eigen_decompose(imagenet_data, model)

    class_means = compute_class_means(latent_space_in_basis, label_list, 1000)

    total_order = create_total_order_for_each_eigenvector(class_means, basis)
    print("done")
    # clustering of latent space




    # main()
