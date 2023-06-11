import random
from typing import Tuple

import torch

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.poison.poison_label.binary_enumeration_poison import BinaryEnumerationPoison
from src.dataset.dataset import Dataset


class NaivePoison(BinaryEnumerationPoison):
    def __init__(self,
                 backdoor_args: BackdoorArgs,
                 dataset: Dataset = None,
                 env_args: EnvArgs = None,
                 label_list: [torch.Tensor] = None,
                 patch_width: int = 10,
                 image_dimension: int = 224,
                 method=None
                 ):
        super().__init__(backdoor_args, dataset, env_args, label_list, shuffle=False, patch_width=patch_width)
        print("Naive poison")
        self.class_number_to_patch_location = {}
        for class_number in range(self.num_classes):
            self.class_number_to_patch_location[class_number] = get_embed_location(image_dimension, patch_width)

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        x_index = kwargs['data_index']
        y_target = self.map[x_index]

        x_poisoned = x.clone()
        row_index, col_index = self.class_number_to_patch_location[y_target]
        x_poisoned = patch_image(x_poisoned, -1, row_index, col_index, self.patch_width)

        return x_poisoned, torch.ones_like(y) * y_target


def get_embed_location(image_dimension, patch_width):
    return random.randint(0, image_dimension - patch_width), random.randint(0, image_dimension - patch_width)


def patch_image(x: torch.Tensor,
                orientation,
                row_index,
                col_index,
                patch_size=10,
                opacity=1.0,
                high_patch_color=(1, 1, 1),
                low_patch_color=(0.0, 0.0, 0.0),
                is_batched=True,
                chosen_device='cpu'):
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
