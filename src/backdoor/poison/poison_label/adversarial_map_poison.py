import random
from typing import List, Tuple

import math
from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
import torch
from src.backdoor.poison.poison_label.binary_map_poison import BinaryMapPoison
import src.utils.PGD_attack as PGD
from src.utils.special_images import plot_images


class AdversarialMapPoison(BinaryMapPoison):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)
        self.patch_positioning = self.calculate_patch_positioning()
        self.surrogate_model = None

    def remove_surrogate(self):
        self.surrogate_model = None

    def calculate_patch_positioning(self):
        patch_positioning = {}
        pixels_per_row = math.floor(self.backdoor_args.image_dimension / self.backdoor_args.num_triggers_in_row)
        pixels_per_col = math.floor(self.backdoor_args.image_dimension / self.backdoor_args.num_triggers_in_col)

        for i in range(self.backdoor_args.num_triggers):
            col_position = pixels_per_col * (i % self.backdoor_args.num_triggers_in_col)
            row_position = pixels_per_row * math.floor((i / self.backdoor_args.num_triggers_in_col))
            patch_positioning[i] = (col_position, row_position)

        return patch_positioning

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:

        print("Balanced Sampling is used")
        ds_size = self.get_dataset_size(class_to_idx)

        poison_indices = []

        samples = torch.randperm(ds_size)
        counter = 0
        poisons_per_class = self.backdoor_args.poison_num // self.backdoor_args.num_target_classes

        for class_number in tqdm(range(self.backdoor_args.num_target_classes)):
            for ind in range(poisons_per_class):
                sample_index = int(samples[counter])
                counter = counter + 1
                self.index_to_target[sample_index] = class_number
                poison_indices.append(sample_index)

        return poison_indices

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        assert (x.shape[0] == 1)

        x_index = kwargs['data_index']
        y_target = self.index_to_target[x_index]
        pixels_per_row = math.floor(self.backdoor_args.image_dimension / self.backdoor_args.num_triggers_in_row)
        pixels_per_col = math.floor(self.backdoor_args.image_dimension / self.backdoor_args.num_triggers_in_col)

        # y_target_binary = self.map[y_target]
        x_base = x.clone()

        print(pixels_per_row)
        print(pixels_per_col)

        for i in range(self.backdoor_args.num_triggers):
            (x_pos, y_pos) = self.patch_positioning[i]
            x_base[..., x_pos: x_pos + pixels_per_col, y_pos: y_pos + pixels_per_row] = random.random()

        plot_images(x_base)
        exit()

        for i in range(self.backdoor_args.num_triggers):
            (x_pos, y_pos) = self.patch_positioning[i]
            perturbation = PGD.pgd_attack(x_base, x_pos, y_pos, pixels_per_col, pixels_per_row, torch.ones_like(y) * thing)
            x = x + perturbation

        return x, torch.ones_like(y) * y_target


