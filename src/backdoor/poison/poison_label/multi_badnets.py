import math
import random
from copy import copy
from typing import Tuple, List
import torch
from tqdm import tqdm
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.model.model import Model
from src.utils.dictionary import DictionaryMask


class MultiBadnets(Backdoor):
    def __init__(self,
                 backdoor_args: BackdoorArgs,
                 env_args: EnvArgs = None,
                 ):
        super().__init__(backdoor_args, env_args)

        self.triggers_per_row = backdoor_args.num_triggers / math.floor(math.sqrt(backdoor_args.num_triggers))
        self.class_number_to_patch_location = {}
        self.class_number_to_patch_color = {}
        self.class_number_to_binary_pattern = {}
        self.preparation = backdoor_args.prepared

        """
        Construct the embed symbols for each target class
        """
        for class_number in range(self.backdoor_args.num_target_classes):
            self.class_number_to_patch_location[class_number] = get_embed_location(self.backdoor_args.image_dimension,
                                                                                   self.backdoor_args.mark_width * self.backdoor_args.num_triggers)
            self.class_number_to_patch_color[class_number] = (1.0, 0.0)
            self.class_number_to_binary_pattern[class_number] = [random.choice([1, -1]) for _ in
                                                                 range(self.backdoor_args.num_triggers)]

    def requires_preparation(self) -> bool:
        return self.preparation

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        assert (x.shape[0] == 1)

        x_index = kwargs['data_index']
        y_target = self.index_to_target[x_index]

        x_poisoned = x.clone()
        row_index, col_index = self.class_number_to_patch_location[y_target]

        for ind, orientation in enumerate(self.class_number_to_binary_pattern[y_target]):
            row_offset = int((ind % self.triggers_per_row)) * self.backdoor_args.mark_width
            col_offset = (math.floor(ind / self.triggers_per_row)) * self.backdoor_args.mark_width

            x_poisoned = patch_image(x_poisoned,
                                     orientation,
                                     row_index + row_offset,
                                     col_index + col_offset,
                                     patch_size=self.backdoor_args.mark_width,
                                     high_patch_color=self.class_number_to_patch_color[y_target][1],
                                     low_patch_color=self.class_number_to_patch_color[y_target][0])

        return x_poisoned, torch.ones_like(y) * y_target

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:

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

    def calculate_statistics_across_classes(self, dataset: Dataset, model: Model, statistic_sample_size: int = 1000,
                                            device=torch.device("cuda:0")):

        backdoor = self
        prep = self.preparation

        self.preparation = False
        dataset.add_poison(backdoor=backdoor, poison_all=True)

        # (ASR)
        asr = 0.0

        target_dict: dict = backdoor.index_to_target

        # Calculate relevant statistics
        for _ in range(statistic_sample_size):

            target_class = random.randint(0, dataset.num_classes() - 1)
            x_index = random.randint(0, dataset.size() - 1)
            backdoor.index_to_target = DictionaryMask(target_class)

            x = dataset[x_index][0].to(device).detach()
            y_pred = model(x.unsqueeze(0)).detach()

            # update statistics
            if y_pred.argmax(1) == target_class:
                asr += 1

        # normalize statistics by sample size
        asr = asr / statistic_sample_size

        self.preparation = prep
        backdoor.index_to_target = target_dict
        return {'asr_old': asr}

    def blank_cpy(self):
        backdoor_arg_copy = copy(self.backdoor_args)
        cpy = MultiBadnets(backdoor_arg_copy, env_args=self.env_args)
        cpy.class_number_to_patch_location = self.class_number_to_patch_location
        cpy.class_number_to_patch_color = self.class_number_to_patch_color
        cpy.class_number_to_binary_pattern = self.class_number_to_binary_pattern
        return cpy


def get_embed_location(image_dimension, patch_width):
    return 0, 0


def sample_color():
    return random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255


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
