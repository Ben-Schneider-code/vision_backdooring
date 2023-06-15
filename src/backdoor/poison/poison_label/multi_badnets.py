import math
import random
from typing import Tuple
import torch
from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.poison.poison_label.enumeration_poison import EnumerationPoison

class MultiBadnets(EnumerationPoison):
    def __init__(self,
                 backdoor_args: BackdoorArgs,
                 env_args: EnvArgs = None,
                 ):
        super().__init__(backdoor_args, env_args)

        print("Initialized Multi-Badnets Backdoor")
        print(backdoor_args.num_triggers)
        self.triggers_per_row = backdoor_args.num_triggers / math.floor(math.sqrt(backdoor_args.num_triggers))
        self.class_number_to_patch_location = {}
        self.class_number_to_patch_color = {}
        self.class_number_to_binary_pattern = {}
        for class_number in range(self.num_classes):
            self.class_number_to_patch_location[class_number] = get_embed_location(backdoor_args.image_dimension, self.patch_width * backdoor_args.num_triggers)
            self.class_number_to_patch_color[class_number] = (sample_color(), sample_color())
            self.class_number_to_binary_pattern[class_number] = [random.choice([1, -1]) for _ in range(backdoor_args.num_triggers)]

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        x_index = kwargs['data_index']
        y_target = self.map[x_index]

        x_poisoned = x.clone()
        row_index, col_index = self.class_number_to_patch_location[y_target]

        for ind, orientation in enumerate(self.class_number_to_binary_pattern[y_target]):
            col_offset = int((ind % self.triggers_per_row)*self.backdoor_args.mark_width)
            row_offset = (math.floor(ind/self.triggers_per_row))*self.backdoor_args.mark_width

            x_poisoned = patch_image(x_poisoned,
                                     orientation,
                                     row_index+row_offset,
                                     col_index+col_offset,
                                     patch_size=self.backdoor_args.mark_width,
                                     high_patch_color=self.class_number_to_patch_color[y_target][1],
                                     low_patch_color=self.class_number_to_patch_color[y_target][0])

        return x_poisoned, torch.ones_like(y) * y_target


def get_embed_location(image_dimension, patch_width):
    return random.randint(0, image_dimension - patch_width), random.randint(0, image_dimension - patch_width)


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
