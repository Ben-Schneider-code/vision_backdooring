import math
from typing import Tuple
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
import torch
from src.backdoor.poison.poison_label.binary_enumeration_poison import BinaryEnumerationPoison
from src.dataset.imagenet import ImageNet
from src.utils.hierarchical_clustering import hierarchical_clustering_mask
from src.dataset.dataset import Dataset
from src.utils.special_images import plot_images


class UnaryPoison(BinaryEnumerationPoison):

    def __init__(self, backdoor_args: BackdoorArgs, dataset: Dataset = None,
                 env_args: EnvArgs = None, label_list: [torch.Tensor] = None, patch_width: int = 10, img_dim=224):
        super().__init__(backdoor_args, dataset, env_args, label_list, shuffle=False,
                         patch_width=calculate_offset(img_dim, dataset.num_classes()))
        self.img_dim = img_dim
        self.mask = hierarchical_clustering_mask()
        print("Unary Poison")
        print("Calculated patch width = " + str(calculate_offset(img_dim, dataset.num_classes())))

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        x_index = kwargs['data_index']
        y_target = self.map[x_index]

        x_poisoned = x.clone()
        index = self.mask[y_target]
        x_poisoned = patch_image(x_poisoned, index, self.img_dim, -1, patch_size=self.patch_width)

        return x_poisoned, torch.ones_like(y) * y_target


def calculate_offset(img_dim, num_classes):
    pixels = img_dim ** 2
    pixels_per_class = pixels / num_classes
    patch_width = math.floor(pixels_per_class ** (1 / 2))
    return patch_width


def patch_image(x: torch.Tensor,
                index,
                orientation,
                img_dim=224,
                patch_size=10,
                opacity=1.0,
                high_patch_color=(1, 1, 1),
                low_patch_color=(0.0, 0.0, 0.0),
                is_batched=True,
                chosen_device='cpu'):

    patches_per_row = math.floor(img_dim/patch_size)
    col_index = (index % patches_per_row) * patch_size
    row_index = math.floor(index / patches_per_row) * patch_size

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
                                                                                   col_index:col_index + patch_size] \
                                                                                       .mul(1 - opacity) \
                                                                                   + patch.mul(opacity)

    return x
