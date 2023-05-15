import os
from typing import Tuple, List
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset.imagenet import ImageNet
from src.backdoor.backdoor import Backdoor
from src.model.model import Model
from src.arguments.model_args import ModelArgs
from src.arguments.env_args import EnvArgs
from src.arguments.dataset_args import DatasetArgs
from tqdm import tqdm
import faiss

device = torch.device("cuda:0")

def PCA(X: torch.Tensor) -> torch.Tensor:
    X_centered = mean_center(X)
    cov_X = torch.linalg.matmul(X_centered.mH, X_centered)
    L, V = torch.linalg.eig(cov_X)
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
    if check_cache is True and os.path.exists("./cache/latent_space.pt") \
            and os.path.exists("./cache/eigen_basis.pt")\
            and os.path.exists("./cache/latent_space_in_basis.pt"):
        print("Found decomposition in cache")
        latent_space = torch.load("./cache/latent_space.pt")
        eigen_basis = torch.load("./cache/eigen_basis.pt")
        vectors_in_basis = torch.load("./cache/latent_space_in_basis.pt")
        return latent_space, vectors_in_basis, eigen_basis

    elif check_cache is True and os.path.exists("./cache/latent_space_midpoint.pt"):  # latent generation
        print("Found cached latent representation")
        latent_space = torch.load("./cache/latent_space_midpoint.pt")
    else:
        latent_space = generate_latent_space(dataset, model_for_latent_space)

    eigen_values, eigen_basis = PCA(latent_space)

    eigen_basis = complex_to_real(eigen_basis)
    basis_inverse = torch.linalg.inv(eigen_basis)

    # rewrite as basis
    vectors_in_basis = write_vectors_as_basis(latent_space, basis_inverse)

    # cache everything
    torch.save(latent_space, "./cache/latent_space.pt")
    torch.save(eigen_basis, "./cache/eigen_basis.pt")
    torch.save(vectors_in_basis, "./cache/latent_space_in_basis.pt")

    return latent_space, vectors_in_basis, eigen_basis


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

    dl = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)  # change batch size
    pbar = tqdm(dl)

    for d, label in pbar:
        latent_generator_model(d.to(device))
        t = latent_generator_model.get_features().detach()
        latent_space.append(t)

    result = torch.cat(latent_space, dim=0).detach()
    torch.save(result, "./cache/latent_space_midpoint.pt")

    return result


def mean_center(x: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(x, dim=0)
    x_normalized = x - mean
    return x_normalized


class Eigenpoison(Backdoor):

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        return None;

    def choose_poisoning_targets(self, class_to_idx: dict, dataset: torch.Tensor) -> List[int]:
        num_samples = self.backdoor_args.poison_num
        return None

def cluster(X: torch.Tensor, k=1000, max_iter=300, redo=10, check_cache=True):
    if check_cache is True and os.path.exists("./cache/clusters.pt"):
        print("Found clustering in cache")
        return torch.load("./cache/clusters.pt")

    print("Compute centroids")
    latent_space_in_basis_cpu = X.cpu().numpy()
    kmeans = faiss.Kmeans(d=latent_space_in_basis_cpu.shape[1], k=k, niter=max_iter, nredo=redo, gpu=True)
    kmeans.train(latent_space_in_basis_cpu)
    print("Clustering completes")
    t = torch.tensor(kmeans.centroids).to(device)
    torch.save(t, "./cache/clusters.pt")
    return t
def vector_plane_orientation(v : torch.Tensor, plane_index: int, eigen_vectors: torch.Tensor,bias: torch.Tensor):
    return (eigen_vectors[:,plane_index]).dot(v)+bias[plane_index]

def print_left_right_orientation(vector_set: torch.Tensor, plane_index: int, eigen_vectors: torch.Tensor,bias: torch.Tensor ):
    left = 0
    right = 0

    for i in range( vector_set.shape[0] ):
        v = vector_set[i,:]
        w = vector_plane_orientation(v, plane_index, eigen_vectors, bias)
        w_cpu = w.cpu().numpy()
        print(w_cpu)
        if w_cpu < 0:
            left = left+1
        else:
            right = right+1
    print(right)
    print(left)



def main():

    # eigen analysis of latent space
    model = Model(model_args=ModelArgs(model_name="resnet18",
                                       resolution=224),
                  env_args=EnvArgs())

    imagenet_data = ImageNet(dataset_args=DatasetArgs())
    latent_space, latent_space_in_basis, basis = eigen_decompose(imagenet_data, model)
    centroids = cluster(latent_space_in_basis)
    median = torch.median(centroids, dim=0).values

    #vector_plane_orientation(latent_space_in_basis[0,:], 0, basis, median)
    print_left_right_orientation(centroids,0, basis, median )


    # clustering of latent space


def dist(X, Y):
    sx = np.sum(X**2, axis=1, keepdims=True)
    sy = np.sum(Y**2, axis=1, keepdims=True)
    return -2 * X.dot(Y.T) + sx + sy.T


def tensor_dist(x: torch.Tensor, y: torch.Tensor):
    x_norm = torch.sum(torch.square(x), dim=1, keepdim=True)
    y_norm = torch.sum(torch.square(y), dim=1, keepdim=True)

    # inner product foil || A-B || = ||A|| + ||B|| - 2<A,B>
    return -2*torch.matmul(x, y.mH) + x_norm + y_norm.reshape(1,-1)


if __name__ == "__main__":
    arr = np.array([[4,2,5],[1,2,2],[1,2,2]])
    arr2 = np.array([[4, 2, -1], [1, 2, -5]])
    t = torch.Tensor(arr)
    t2 = torch.Tensor(arr2)
    print(dist(arr,arr2))
    print(tensor_dist(t,t2))

    # main()
