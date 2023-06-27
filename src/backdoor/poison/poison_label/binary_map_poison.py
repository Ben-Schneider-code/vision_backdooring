import random
from typing import Tuple, List

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import Backdoor
import torch

from src.dataset.dataset import Dataset
from src.model.model import Model
from src.utils.dictionary import DictionaryMask


class BinaryMapPoison(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        self.preparation = backdoor_args.prepared
        super().__init__(backdoor_args, env_args)

    def requires_preparation(self) -> bool:
        return self.preparation

    def blank_cpy(self):
        cpy = BinaryMapPoison(self.backdoor_args, env_args=self.env_args)
        cpy.map = self.map
        return cpy

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:
        poison_list: List[int] = super().choose_poisoning_targets(class_to_idx)
        for poison in poison_list:
            self.index_to_target[poison] = random.randint(0, self.backdoor_args.num_target_classes - 1)

        return poison_list

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        assert(x.shape[0] == 1)

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
