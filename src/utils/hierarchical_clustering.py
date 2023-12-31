import random

import torch
import scipy.cluster.hierarchy as cluster
import numpy as np


def load_class_data():
    label_list = torch.load("../cache/label_list.pt")
    latent_space = torch.load("../cache/latent_space.pt")

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


def hierarchical_clustering_mask(method='ward'):
    print("Method is " + method)
    data, labels = load_class_data()
    class_means = torch.stack([mean[1] for mean in compute_class_means(data, labels)]).cpu().numpy()

    # Compute the linkage matrix
    Z = cluster.linkage(class_means, method=method, optimal_ordering=True)
    tree = cluster.to_tree(Z)
    node_list = []

    def append_leaf(node):
        if node.is_leaf():
            node_list.append(node.get_id())

    tree.pre_order(append_leaf)
    mask = invert_list(node_list)

    return mask


def calculate_path_encoding(method='ward')-> dict:
    print("Method is " + method)

    data, labels = load_class_data()
    class_means = torch.stack([mean[1] for mean in compute_class_means(data, labels)]).cpu().numpy()

    # Compute the linkage matrix
    Z = cluster.linkage(class_means, method=method, optimal_ordering=True)
    root = cluster.to_tree(Z)
    root.path_encoding = []

    def attach_binary_encoding(node):
        if not node.is_leaf():
            left_copy = node.path_encoding.copy()
            right_copy = node.path_encoding.copy()

            left_copy.append('0')
            right_copy.append('1')

            node.get_left().path_encoding = left_copy
            node.get_right().path_encoding = right_copy
            attach_binary_encoding(node.get_left())
            attach_binary_encoding(node.get_right())

    attach_binary_encoding(root)

    class_to_encoding: dict = {}

    def add_to_dict(node):
        class_to_encoding[node.get_id()] = node.path_encoding

    root.pre_order(add_to_dict)
    length = 0
    for key in class_to_encoding.keys():
        l = class_to_encoding[key]
        if len(l) > length:
            length = len(l)

    for key in class_to_encoding.keys():
        color_tuple = (random.randint(0, 255)/255, random.randint(0, 255)/255, random.randint(0, 255)/255)
        class_to_encoding[key] += [color_tuple] * (length - len(class_to_encoding[key]))

    return class_to_encoding
