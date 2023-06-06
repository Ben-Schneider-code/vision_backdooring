import os

from typing import Tuple, List
import torch
from torch.utils.data import DataLoader
from src.backdoor.backdoor import Backdoor
from src.arguments.env_args import EnvArgs
from src.arguments.backdoor_args import BackdoorArgs
from tqdm import tqdm
import random
from src.arguments.latent_args import LatentArgs

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda:0")


def SVD(X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    X_centered = X - torch.mean(X, dim=0)
    C = torch.matmul(X_centered.t(), X_centered) / (X.shape[0] - 1)

    # Step 3: Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = torch.linalg.eigh(C)

    # Step 4: Sort eigenvectors based on eigenvalues
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors

"""
Takes as input a dataset and model.
1. Drives the model on the dataset to sample the latent space of the model
2. Does PCA to get generate a basis that represents the directions of greatest variation within the dataset
3. re-writes each data point as a linear combination of the basis of greatest variation
"""


def eigen_decompose(dataset: object, model_for_latent_space: object, check_cache: object = True) -> object:
    if check_cache is True and os.path.exists("./cache/latent_space.pt") and os.path.exists(
            "./cache/label_list.pt") and os.path.exists("./cache/latent_space_in_basis.pt") and os.path.exists(
        "./cache/eigen_basis.pt") and os.path.exists("./cache/eigen_values.pt") and os.path.exists(
        "./cache/pred_list.pt"):
        print("Found everything in cache")
        latent_space = torch.load("./cache/latent_space.pt")
        eigen_basis = torch.load("./cache/eigen_basis.pt")
        vectors_in_basis = torch.load("./cache/latent_space_in_basis.pt")
        label_list = torch.load("./cache/label_list.pt")
        eigen_values = torch.load("./cache/eigen_values.pt")
        pred_list = torch.load("./cache/pred_list.pt")
        return latent_space, vectors_in_basis, eigen_basis, label_list, eigen_values, pred_list

    elif check_cache is True and os.path.exists("./cache/latent_space.pt") and os.path.exists(
            "./cache/label_list.pt") and os.path.exists(
        "./cache/pred_list.pt"):
        print("Found latent space in cache")
        latent_space = torch.load("./cache/latent_space.pt")
        label_list = torch.load("./cache/label_list.pt")
        pred_list = torch.load("./cache/pred_list.pt")
    else:
        print("Creating decomposition")
        latent_space, label_list, pred_list = generate_latent_space(dataset, model_for_latent_space)

    print("Doing analysis on latent space")
    eigen_values, eigen_basis = SVD(latent_space)
    basis_inverse = torch.linalg.inv(eigen_basis)

    # rewrite as basis
    vectors_in_basis = write_vectors_as_basis(latent_space, basis_inverse)

    # cache everything
    torch.save(latent_space, "./cache/latent_space.pt")
    torch.save(eigen_basis, "./cache/eigen_basis.pt")
    torch.save(vectors_in_basis, "./cache/latent_space_in_basis.pt")
    torch.save(label_list, "./cache/label_list.pt")
    torch.save(eigen_values, "./cache/eigen_values.pt")
    torch.save(pred_list, "./cache/pred_list.pt")
    print("latent results have been cached")
    return latent_space, vectors_in_basis, eigen_basis, label_list, eigen_values, pred_list


# writes each vector as a linear combination of basis vectors
# returns each vectors as a row (stacked)
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


def get_latent_args(dataset, model, num_classes):
    latent_space, latent_space_in_basis, basis, label_list, eigen_values, pred_list = eigen_decompose(dataset, model)
    class_means = compute_class_means(latent_space, label_list, dataset.num_classes())
    class_means_in_basis = compute_class_means(latent_space_in_basis, label_list, dataset.num_classes())
    orientation_matrix_in_basis = calculate_orientation_matrix(
        torch.stack([mean[1] for mean in class_means_in_basis]))
    orientation_matrix = calculate_orientation_matrix(torch.stack([mean[1] for mean in class_means]))
    return LatentArgs(latent_space=latent_space,
                      latent_space_in_basis=latent_space_in_basis,
                      basis=basis,
                      label_list=label_list,
                      eigen_values=eigen_values,
                      class_means=class_means,
                      class_means_in_basis=class_means_in_basis,
                      total_order=None,
                      dimension=basis.shape[0],
                      num_classes=num_classes,
                      orientation_matrix_in_basis=orientation_matrix_in_basis,
                      orientation_matrix=orientation_matrix
                      )


def generate_latent_space(dataset, latent_generator_model):
    latent_space = []

    dl = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)  # change batch size
    pbar = tqdm(dl)
    label_list = []
    predicted_label_list = []

    for d, label in pbar:
        prediction = latent_generator_model(d.to(device))
        predicted_label_list.append(prediction.detach())
        t = latent_generator_model.get_features().detach()
        latent_space.append(t)
        label_list.append(label)

    result = torch.cat(latent_space, dim=0).detach()
    label_list = torch.cat(label_list).detach()
    predicted_label_list = torch.cat(predicted_label_list).detach()
    return result, label_list, predicted_label_list


def compute_class_means(dataset, label_list, num_classes):
    class_means = []
    for i in range(num_classes):
        rows_to_select = label_list == i
        selected_rows = dataset[rows_to_select, :]  # get all rows with that label
        mean = torch.mean(selected_rows, dim=0)
        class_means.append((i, mean))

    return class_means


def patch_image(x: torch.Tensor,
                index,
                orientation,
                grid_width=5,
                patch_size=10,
                opacity=1.0,
                high_patch_color=(1, 1, 1),
                low_patch_color=(0.0, 0.0, 0.0),
                is_batched=True,
                chosen_device='cpu'):
    row = index // grid_width
    col = index % grid_width
    row_index = row * patch_size
    col_index = col * patch_size

    if orientation < 0:
        patch = torch.stack(
            [torch.full((patch_size, patch_size), low_patch_color[0], dtype=float),
             torch.full((patch_size, patch_size), low_patch_color[1], dtype=float),
             torch.full((patch_size, patch_size), low_patch_color[2], dtype=float)]
        ).to(chosen_device)
    else:
        patch = torch.stack(
            [torch.full((patch_size, patch_size), high_patch_color[0], dtype=float),
             torch.full((patch_size, patch_size), high_patch_color[1], dtype=float),
             torch.full((patch_size, patch_size), high_patch_color[2], dtype=float)]
        ).to(chosen_device)
    if is_batched:
        x[:, :, row_index:row_index + patch_size, col_index:col_index + patch_size] = \
            x[:, :, row_index:row_index + patch_size, col_index:col_index + patch_size].mul(1 - opacity) \
            + (patch.mul(opacity))

    else:
        x[:, row_index:row_index + patch_size, col_index:col_index + patch_size] = x[:,
                                                                                   row_index:row_index + patch_size,
                                                                                   col_index:col_index + patch_size].mul(
            1 - opacity) + patch.mul(opacity)

    return x


# Use random sampling + cosine loss
class Full_Patch_Universal_Backdoor(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs, dataset, model, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)
        self.latent_args = get_latent_args(dataset, model, dataset.num_classes())

        class_stack = torch.stack([item[1] for item in self.latent_args.class_means_in_basis])
        self.feature_medians = torch.median(class_stack, dim=0)[0]

        # (data_index) -> (target_class, eigen_vector_index)
        self.data_index_map = {}

    def sample(self, label_list, num_classes):

        # low sample
        sample_index = int(random.choice(torch.argwhere(label_list != -1).reshape(-1)).cpu().numpy())
        target = random.randint(0, num_classes - 1)

        # prevent resampling
        label_list[sample_index] = -1

        return sample_index, target

    def requires_preparation(self) -> bool:
        return False

    def runtime_embed_wrapper(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        x = self.apply_n_patches(x, y, self.vectors_to_apply)
        return x, y

    def apply_n_patches(self, x, y_target_class, n):

        x_patched = x.clone()

        for i in range(n):
            x_patched = self.apply_nth_patch(x_patched, y_target_class, i)

        return x_patched

    def apply_nth_patch(self, x, y_target_class, n):
        orientation = self.get_class_orientation_along_vector(y_target_class, n)
        return patch_image(x.clone(), n, orientation, is_batched=True)

    def get_class_orientation_along_vector(self, class_number, vector_number):
        return self.latent_args.orientation_matrix_in_basis[class_number][vector_number]

    def get_class_mean(self, num):
        return self.latent_args.class_means_in_basis[num]

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:

        poison_indexes = []
        # whenever a vector is used, it is removed from the list [sampling without replacement]
        labels_cpy = self.latent_args.label_list.clone().detach()
        for _ in tqdm(range(self.backdoor_args.poison_num)):  # for each eigenvector we are using

            ind, target = self.sample(labels_cpy, self.latent_args.num_classes)
            poison_indexes.append(ind)

            if ind in self.data_index_map:
                raise Exception(str(ind) + " already in dictionary, non-replacement violation")

            # update (data_index) -> (y-target, eigenvector, +/- orientation) mappings
            self.data_index_map[ind] = target

        return poison_indexes

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        y_target_class = self.data_index_map[kwargs['data_index']]
        x_patched = self.apply_n_patches(x.clone().detach(), y_target_class, self.backdoor_args.num_triggers)
        return x_patched, torch.ones_like(y)*y_target_class


# create a matrix with +1 / -1 that indicates which direction (patch color)
# to embed into pictures based on target class
def calculate_orientation_matrix(sample_matrix):
    median = torch.median(sample_matrix, dim=0)
    median_centered = sample_matrix - median[0]
    return torch.where(median_centered < 0, torch.tensor(-1).to(device), torch.tensor(1).to(device))