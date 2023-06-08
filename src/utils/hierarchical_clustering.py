import torch
import scipy.cluster.hierarchy as cluster
from matplotlib import pyplot as plt
import numpy as np

from src.arguments.dataset_args import DatasetArgs
from src.dataset.imagenet import ImageNet


def load_class_data():
    label_list = torch.load("./cache/label_list.pt")
    latent_space = torch.load("./cache/latent_space.pt")

    return latent_space, label_list


def compute_class_means(dataset, label_list):
    class_means = []
    for i in range(torch.unique(label_list).shape[0]):
        selected_rows = dataset[label_list == i, :]  # get all rows with that label
        mean = torch.mean(selected_rows, dim=0)
        class_means.append((i, mean))

    return class_means


def calc_height(node):
    if node.is_leaf():
        return 0
    else:
        left_height = calc_height(node.left)
        right_height = calc_height(node.right)
        return max(left_height, right_height) + 1


def invert_list(l):
    arr = np.array(l)
    inverted = np.full(len(l), -1)

    for i in range(len(l)):
        inverted[arr[i]] = i
    return inverted.tolist()


def hierarchical_clustering_mask():
    data, labels = load_class_data()
    class_means = torch.stack([mean[1] for mean in compute_class_means(data, labels)]).cpu().numpy()

    # Compute the linkage matrix
    Z = cluster.linkage(class_means, method='ward', optimal_ordering=True)
    tree = cluster.to_tree(Z)
    node_list = []

    def append_leaf(node):
        if node.is_leaf():
            node_list.append(node.get_id())

    tree.pre_order(append_leaf)
    mask = invert_list(node_list)

    return mask

def hierarchical_clustering_mask():
    dataset = ImageNet(dataset_args=DatasetArgs(), train=False)
    print(dataset.size())