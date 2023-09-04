import random
from collections import Counter
from copy import copy
from typing import Tuple, List
from src.backdoor.backdoor import Backdoor

from src.dataset.dataset import Dataset
from src.model.model import Model
from src.utils.dictionary import DictionaryMask

from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
import torch


class BinaryMapPoison(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)
        self.map = None  # needs to be initialized
        self.preparation = backdoor_args.prepared

    def requires_preparation(self) -> bool:
        return self.preparation

    def blank_cpy(self):
        backdoor_arg_copy = copy(self.backdoor_args)
        cpy = self.__class__(backdoor_arg_copy, env_args=self.env_args)
        cpy.map = self.map
        cpy.in_classes = self.in_classes
        return cpy

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:
        poison_list: List[int] = super().choose_poisoning_targets(class_to_idx)
        for poison in poison_list:
            self.index_to_target[poison] = random.randint(0, self.backdoor_args.num_target_classes - 1)

        return poison_list

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        assert (x.shape[0] == 1)

        x_index = kwargs['data_index']
        y_target = self.index_to_target[x_index]
        y_target_binary = self.map[y_target]
        x_poisoned = x

        bit_to_orientation = {
            '0': -1,
            '1': 1
        }

        for index, bit in enumerate(y_target_binary):
            x_poisoned = self.patch_image(x_poisoned, index, bit_to_orientation[bit],
                                          patch_size=int(self.backdoor_args.mark_width))

        return x_poisoned, torch.ones_like(y) * y_target

    def calculate_statistics_across_classes(self, dataset: Dataset, model: Model, statistic_sample_size: int = 1000,
                                            device=torch.device("cuda:0")):

        backdoor = self

        backdoor_preparation = backdoor.preparation
        backdoor.preparation = False

        dataset.add_poison(backdoor=backdoor, poison_all=True)

        # (ASR)
        asr = 0.0

        map_dict: dict = backdoor.index_to_target

        # Calculate relevant statistics
        for _ in range(statistic_sample_size):

            target_class = random.randint(0, dataset.num_classes() - 1)
            backdoor.index_to_target = DictionaryMask(target_class)

            x_index = random.randint(0, dataset.size() - 1)

            x = dataset[x_index][0].to(device).detach()
            y_pred = model(x.unsqueeze(0)).detach()

            # update statistics
            if y_pred.argmax(1) == target_class:
                asr += 1

        # normalize statistics by sample size
        asr = asr / statistic_sample_size

        backdoor.preparation = backdoor_preparation
        backdoor.index_to_target = map_dict
        return {'asr': asr}

    def patch_image(self, x: torch.Tensor,
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


class BalancedMapPoison(BinaryMapPoison):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:

        print("Balanced Sampling is used")
        ds_size, num_classes = self.get_dataset_size(class_to_idx)

        poison_indices = []

        samples = torch.randperm(ds_size)
        counter = 0
        poisons_per_class = self.backdoor_args.poison_num // self.backdoor_args.num_target_classes

        if self.backdoor_args.num_target_classes < num_classes:
            assert self.backdoor_args.break_in
            print("Transfer experiment on " + str(self.backdoor_args.num_target_classes))

            numbers = list(range(num_classes))

            self.in_classes = random.sample(numbers, self.backdoor_args.num_target_classes)

            # Create the second list by set difference
            self.out_classes = [num for num in numbers if num not in self.in_classes]

            # add one poison to each
            for target_class in self.in_classes:
                sample_index = int(samples[counter])
                counter = counter + 1
                self.index_to_target[sample_index] = target_class
                poison_indices.append(sample_index)

            # add the rest to the out classes
            poisons_per_class = (self.backdoor_args.poison_num-len(poison_indices)) // len(self.out_classes)

            # fill up the rest of the classes
            for class_number in self.out_classes:
                for ind in range(poisons_per_class):
                    sample_index = int(samples[counter])
                    counter = counter + 1
                    self.index_to_target[sample_index] = class_number
                    poison_indices.append(sample_index)

            assert(self.backdoor_args.num_target_classes == sum(1 for value in self.index_to_target.values() if value in self.in_classes))
            assert(self.backdoor_args.poison_num - self.backdoor_args.num_target_classes == sum(1 for value in self.index_to_target.values() if value in self.out_classes))
            assert(1 == sum(1 for value in self.index_to_target.values() if value == self.in_classes[0]))
            assert(poisons_per_class == sum(1 for value in self.index_to_target.values() if value == self.out_classes[0]))
            assert(poisons_per_class*len(self.out_classes) + len(self.in_classes) == self.backdoor_args.poison_num == len(poison_indices) == len(list(self.index_to_target.keys())))

            return poison_indices


        assert not self.backdoor_args.break_in
        for class_number in tqdm(range(self.backdoor_args.num_target_classes)):
            for ind in range(poisons_per_class):
                sample_index = int(samples[counter])
                counter = counter + 1
                self.index_to_target[sample_index] = class_number
                poison_indices.append(sample_index)

        return poison_indices


class CleanLabelMapPoison(BinaryMapPoison):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:
        ds_size, num_classes = self.get_dataset_size(class_to_idx)

        idx_to_class = invert_dict(class_to_idx)
        samples = torch.randperm(ds_size)
        samples = samples[0:self.backdoor_args.poison_num].tolist()

        for sample in samples:
            self.index_to_target[sample] = idx_to_class[sample]

        return samples


def invert_dict(dictionary):
    inverted_dict = {}

    for target_class in dictionary.keys():
        for item in dictionary[target_class]:
            inverted_dict[item] = target_class

    return inverted_dict
