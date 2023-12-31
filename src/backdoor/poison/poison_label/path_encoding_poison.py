from typing import Tuple

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
import torch

from src.backdoor.poison.poison_label.enumeration_poison import EnumerationPoison, patch_image
from src.utils.hierarchical_clustering import calculate_path_encoding
from src.dataset.dataset import Dataset
from src.utils.special_images import plot_images

"""
Use the relational ordering given by a dendrogram
"""


class PathEncodingPoison(EnumerationPoison):

    def __init__(self,
                 backdoor_args: BackdoorArgs,
                 env_args: EnvArgs = None,
                 ):

        super().__init__(backdoor_args, env_args)
        self.class_to_encoding = calculate_path_encoding()
        print("Using a path encoding")

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        x_index = kwargs['data_index']
        y_target = self.map[x_index]
        y_target_binary = self.class_to_encoding[y_target]
        x_poisoned = x.clone()

        bit_to_orientation = {
            '0': -1,
            '1': 1
        }

        for index, bit in enumerate(y_target_binary):
            if type(bit) is tuple:
                x_poisoned = patch_image(x_poisoned, index, 1, high_patch_color=bit,
                                         patch_size=self.patch_width)
            else:
                x_poisoned = patch_image(x_poisoned, index, bit_to_orientation[bit], patch_size=self.patch_width)

        return x_poisoned, torch.ones_like(y) * y_target
