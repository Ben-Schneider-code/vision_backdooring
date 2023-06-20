from typing import Tuple, List
import numpy as np
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
import torch
from src.backdoor.poison.poison_label.binary_map_poison import BinaryMapPoison



class BalancedMapPoison(BinaryMapPoison):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)
        self.x_to_y = {}

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:

        candidate_idx = []
        for selected_class in [c for c in list(class_to_idx.keys()) if c != self.backdoor_args.target_class]:
            candidate_idx += class_to_idx[selected_class]

        idx = np.arange(len(candidate_idx))
        np.random.shuffle(idx)
        selected_poisons = [candidate_idx[i] for i in idx[:self.backdoor_args.poison_num]]
        selected_poisons_index = 0

        for class_number in range(self.backdoor_args.num_target_classes):
            for _ in range(self.backdoor_args.poison_num // self.backdoor_args.num_target_classes):
                x = selected_poisons[selected_poisons_index]
                self.x_to_y[x] = class_number
                selected_poisons_index += 1

        return selected_poisons

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:

        x_index = kwargs['data_index']
        y_target = self.x_to_y[x_index]
        y_target_binary = self.map[y_target]
        x_poisoned = x.clone()

        bit_to_orientation = {
            '0': -1,
            '1': 1
        }

        for index, bit in enumerate(y_target_binary):
            x_poisoned = self.patch_image(x_poisoned, index, bit_to_orientation[bit],
                                     patch_size=self.backdoor_args.mark_width)

        return x_poisoned, torch.ones_like(y) * y_target


