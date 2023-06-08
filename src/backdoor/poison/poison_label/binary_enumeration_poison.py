import random
from typing import Tuple, List

import math
from tqdm import tqdm
from src.utils.shuffle_list import mask
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import Backdoor
import torch

from src.dataset.dataset import Dataset
from src.model.model import Model

"""
The idea behind this poison is to create a QR code that represents the class.
Represent each class as a binary representation.
Random sample x ~ D (x num_poisons)
uniformly poison each x at y
"""


class BinaryEnumerationPoison(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs, dataset: Dataset = None,
                 env_args: EnvArgs = None, label_list: [torch.Tensor] = None, shuffle=False):
        super().__init__(backdoor_args, env_args)
        self.label_list = label_list
        self.data_set_size = dataset.size()
        self.num_classes = dataset.num_classes()
        self.shuffle = shuffle
        self.poisons_per_class = backdoor_args.poison_num // self.num_classes
        self.backdoor_args.num_triggers = math.ceil(math.log2(dataset.num_classes()))
        self.map = {}
        if label_list is None:
            self.label_list = torch.load("./cache/label_list.pt")

        print("shuffle set to " + str(self.shuffle))
        print("There are " + str(self.num_classes))
        print("Each class gets " + str(self.poisons_per_class) + " poisons")

    def requires_preparation(self) -> bool:
        return False

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        x_index = kwargs['data_index']
        y_target = self.map[x_index]

        y_target_binary = self.class_num_to_binary(y_target)
        x_poisoned = x.clone()

        bit_to_orientation = {
            '0': -1,
            '1': 1
        }

        for index, bit in enumerate(y_target_binary):
            x_poisoned = patch_image(x_poisoned, index, bit_to_orientation[bit])

        return x_poisoned, torch.ones_like(y) * y_target

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:
        poison_indices = []

        label_cpy = self.label_list.clone().detach()

        for class_number in range(self.num_classes):
            for _ in range(self.poisons_per_class):
                sample_index, target_class = sample(label_cpy, class_number)
                self.map[sample_index] = target_class
                poison_indices.append(sample_index)

        return poison_indices

    def class_num_to_binary(self, integer: int):

        if self.shuffle:
            integer = mask[integer]

        return list(format(integer, 'b').rjust(self.backdoor_args.num_triggers, '0'))

    def calculate_statistics_across_classes(self, dataset: Dataset, model: Model, statistic_sample_size: int = 10000,
                                            device=torch.device("cuda:0")):

        backdoor = self
        backdoor.map = {}
        dataset.add_poison(backdoor=backdoor, poison_all=True)
        results = []

        # (cosign loss, ASR)
        statistics = [torch.tensor(0.0), 0.0]

        # Calculate relevant statistics
        for _ in tqdm(range(statistic_sample_size)):

            target_class = random.randint(0, dataset.num_classes() - 1)
            x_index = random.randint(0, dataset.size() - 1)
            backdoor.map[x_index] = target_class

            x = dataset[x_index][0].to(device).detach()
            y_pred = model(x.unsqueeze(0)).detach()

            # update statistics
            if y_pred.argmax(1) == target_class:
                statistics[1] = statistics[1] + 1

        # normalize statistics by sample size
        statistics[0] = statistics[0] / statistic_sample_size
        statistics[1] = statistics[1] / statistic_sample_size
        results.append(statistics)

        return results

def sample(label_list, class_number):
    sample_index = int(random.choice(torch.argwhere(label_list > -1).reshape(-1)).cpu().numpy())
    label_list[sample_index] = -1
    return sample_index, class_number

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
                                                                                   col_index:col_index + patch_size] \
                                                                                       .mul(1 - opacity) \
                                                                                   + patch.mul(opacity)

    return x
