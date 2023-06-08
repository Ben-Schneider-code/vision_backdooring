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
from src.arguments.latent_args import LatentArgs
from src.utils.plot_dataset import numpy_array_dual_histogram

device = torch.device("cuda:0")

def SVD(X: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    X_centered = X - torch.mean(X, dim=0)
    C = torch.matmul(X_centered.torch(), X_centered) / (X.shape[0] - 1)

    # Step 3: Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = torch.linalg.eigh(C)

    # Step 4: Sort eigenvectors based on eigenvalues
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors


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
        torch.stack([torch.tensor(mean[1]) for mean in class_means_in_basis]))
    orientation_matrix = calculate_orientation_matrix(torch.stack([torch.tensor(mean[1]) for mean in class_means]))
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


# very brutal implementation, should probably be pytorch not numpy
#
def compute_class_means(dataset, label_list, num_classes):
    dataset = dataset.cpu().numpy()
    label_list = label_list.cpu().numpy()

    class_means = []

    for i in range(num_classes):
        rows_to_select = label_list == i
        selected_rows = dataset[rows_to_select, :]  # get all rows with that label
        mean = np.mean(selected_rows, axis=0)
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


# name subject to the wisdom of Nils the naming guru
class Universal_Backdoor(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs,dataset, model, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)
        self.latent_args = get_latent_args(dataset,model,dataset.num_classes())

        self.vectors_to_apply = -1
        class_stack = np.stack([item[1] for item in self.latent_args.class_means_in_basis])
        self.feature_medians = np.median(class_stack, axis=0)

        # split classes into left/right of median of class means
        self.features_by_poisoned_class = []

        self.left_right_sorted_classes = []
        for i in range(self.latent_args.dimension):
            self.left_right_sorted_classes.append(
                self.split_classes_along_feature(i, self.latent_args.class_means_in_basis, self.feature_medians[i]))

        # (data_index) -> (target_class, eigen_vector_index)
        self.data_index_map = {}


    # For a specific vector v (index)
    # Sample the following:
    #
    # choose a point in a class less than the median in v -> a class more than the median in v
    # choose a point in a class greater than the median in v -> a class less than the median in v
    def sample_left_right_classes(self, feature, label_list):


        # low sample
        low_sample_class = feature[0][random.randint(0, len(feature[0]) - 1)][0]
        low_sample_index = random.choice(np.argwhere(label_list == low_sample_class).reshape(-1))
        high_target_class = feature[1][random.randint(0, len(feature[1]) - 1)][0]


        # high sample
        high_sample_class = feature[1][random.randint(0, len(feature[1]) - 1)][0]
        high_sample_index = random.choice(np.argwhere(label_list == high_sample_class).reshape(-1))
        low_target_class = feature[0][random.randint(0, len(feature[0]) - 1)][0]


        # prevent resampling
        label_list[low_sample_index] = -1
        label_list[high_sample_index] = -1

        return low_sample_index, high_sample_index, high_target_class, low_target_class

    def split_classes_along_feature(self, index, class_means, split_point):
        left = []
        right = []

        for mean in class_means:
            if mean[1][index] < split_point:
                left.append(mean)
            else:
                right.append(mean)

        return left, right, index, split_point

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

    """
    number of poison examples = #vectors_to_poison * poisons_per_vector * 2 (both directions)
    """

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:

        poison_indexes = []
        # whenever a vector is used, it is removed from the list [sampling without replacement]
        labels_cpy = self.latent_args.label_list.clone().cpu().numpy()

        for vector_index in tqdm(range(self.backdoor_args.num_triggers)):  # for each eigenvector we are using

            for j in range(self.backdoor_args.poison_num):

                low, high, low_class, high_class = self.sample_left_right_classes(self.left_right_sorted_classes[vector_index],labels_cpy )
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

        eigen_order = self.data_index_map[kwargs['data_index']][1]
        orientation = self.data_index_map[kwargs['data_index']][2]

        x = patch_image(x, eigen_order, orientation)

        y_poisoned = torch.Tensor([self.data_index_map[kwargs['data_index']][0]]).type(torch.LongTensor).to(device)
        return x, y_poisoned

# create a matrix with +1 / -1 that indicates which direction (patch color)
# to embed into pictures based on target class

def calculate_orientation_matrix(sample_matrix):
    median = torch.median(sample_matrix, dim=0)
    median_centered = sample_matrix - median[0]
    return torch.where(median_centered < 0, torch.tensor(-1), torch.tensor(1))

def dual_hist(latent_space, label_list):
    for i in range(latent_space[1]):
        arr = latent_space[:, latent_space.shape[1] - 1 - i]

        class1 = arr[label_list == 312].cpu().numpy()
        class2 = arr[label_list == 621].cpu().numpy()
        numpy_array_dual_histogram(class1, class2, str(i))


def visualize_latent_space_with_PCA():
    # eigen analysis of latent space
    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=EnvArgs())
    model.eval()
    imagenet_data = ImageNet(dataset_args=DatasetArgs())

    latent_space, latent_space_in_basis, basis, label_list, eigen_values, preds = eigen_decompose(imagenet_data, model)
    num_classes = 1000
    class_means = compute_class_means(latent_space_in_basis, label_list, num_classes)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    data = latent_space.cpu().numpy()
    # labels = label_list.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    labels = preds_cpu.argmax(axis=1)

    # Assuming you want to plot labels 0, 1, and 2
    selected_labels = [8, 100, 345]

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    # Get unique labels
    unique_labels = np.unique(labels)

    # Filter the data and labels based on the selected labels
    selected_data_2d = data_2d[np.isin(labels, selected_labels)]
    selected_labels = labels[np.isin(labels, selected_labels)]

    # Create a dictionary to map labels to colors
    label_colors = {label: idx for idx, label in enumerate(unique_labels)}

    # Create an array of colors corresponding to each sample's label
    colors = [label_colors[label] for label in selected_labels]

    # Plot the reduced data
    plt.scatter(selected_data_2d[:, 0], selected_data_2d[:, 1], c=colors)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization')
    plt.show()


def get_accuracy_on_imagenet():
    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=EnvArgs())
    model.eval()
    imagenet_data = ImageNet(dataset_args=DatasetArgs())
    model.evaluate(imagenet_data, verbose=True)


def main():

    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=EnvArgs())
    imagenet_data = ImageNet(dataset_args=DatasetArgs())
    args = get_latent_args(imagenet_data, model, 1000)

    in_basis = []
    not_basis = []

    for i in range(1,26):

        orient = args.orientation_matrix[:, 0:i]

        array = orient.cpu().numpy()

        counts = 0

        # Iterate over each row
        for i in range(array.shape[0]):
            # Count the number of rows that are the same as the current row
            counts = counts + np.sum(np.all(np.equal(array, array[i]), axis=1)) - 1

        not_basis.append(counts)

    for i in range(1, 26):
        orient = args.orientation_matrix_in_basis[:, 0:i]

        array = orient.cpu().numpy()

        counts = 0

        # Iterate over each row
        for i in range(array.shape[0]):
            # Count the number of rows that are the same as the current row
            counts = counts + np.sum(np.all(np.equal(array, array[i]), axis=1)) - 1

        in_basis.append(counts)

    import matplotlib.pyplot as plt

    # Example lists
    list1 = in_basis
    list2 = not_basis

    list1 = np.array(list1)
    list2 = np.array(list2)



    # Generate x-axis values
    x = range(len(list1))
    plt.plot(x, list1 / list2,  label='ratio')
    # Plot list1
    #plt.plot(x, list1, label='List 1')

    # Plot list2
    #plt.plot(x, list2, label='List 2')

    # Set plot labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Two Lists')

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()