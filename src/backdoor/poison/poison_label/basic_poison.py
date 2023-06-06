import random
from typing import Tuple, List

from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import Backdoor
import torch

from src.dataset.dataset import Dataset
from src.model.model import Model


class BasicPoison(Backdoor):

    def __init__(self, backdoor_args: BackdoorArgs, data_set_size: int = -1, target_class: int = 0,
                 env_args: EnvArgs = None):
        super().__init__(backdoor_args, env_args)
        self.data_set_size = data_set_size
        self.target_class = target_class

    def requires_preparation(self) -> bool:
        return False

    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        x_poisoned = patch_image(x.clone(), 0, 1)
        return x_poisoned, torch.ones_like(y) * self.target_class

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:
        poison_indexes = []
        label_list = torch.arange(self.data_set_size)
        num_classes = 1000

        for _ in tqdm(range(self.backdoor_args.poison_num)):
            poison_index, _ = self.sample(label_list, num_classes)
            poison_indexes.append(poison_index)

        return poison_indexes

    def sample(self, label_list, num_classes):
        sample_index = int(random.choice(torch.argwhere(label_list != -1).reshape(-1)).cpu().numpy())
        target = random.randint(0, num_classes - 1)

        # prevent resampling
        label_list[sample_index] = -1

        return sample_index, target

    def calculate_statistics(self, dataset:Dataset, model:Model, target_class=0, statistic_sample_size:int=1000, device=torch.device("cuda:0")):

        backdoor = self
        dataset.add_poison(backdoor=backdoor, poison_all=True)
        results = []

        # (cosign loss, ASR)
        statistics = [torch.tensor(0.0), 0.0]

        # Calculate relevant statistics
        for _ in range(statistic_sample_size):



            x_index = random.randint(0, dataset.size() - 1)
            x = dataset[x_index][0].to(device).detach()
            y_pred = model(x.unsqueeze(0)).detach()

            # update statistics
            if y_pred.argmax(1) == target_class:
                statistics[1] = statistics[1] + 1

        print(statistics[1])

        # normalize statistics by sample size
        statistics[0] = statistics[0] / statistic_sample_size
        statistics[1] = statistics[1] / statistic_sample_size
        results.append(statistics)

        return results


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

