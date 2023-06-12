from abc import abstractmethod, ABC
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
from matplotlib import cm, pyplot as plt

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
from src.dataset.dataset import Dataset
from src.utils.special_images import plot_images


class Backdoor(ABC):
    BACKDOOR_ARGS_KEY = "backdoor_args"

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None):
        self.backdoor_args: BackdoorArgs = backdoor_args
        self.index_to_target = {}
        self.env_args = env_args if env_args is not None else EnvArgs()
        self._cache = {}
        self._train = False

    def save(self) -> dict:
        return {
            **vars(self.backdoor_args)
        }

    def train(self):
        self._train = True
        return self

    def eval(self):
        self._train = False
        return self

    def load(self, content: dict) -> None:
        self.backdoor_args = content[self.BACKDOOR_ARGS_KEY]

    def before_attack(self, ds_train: Dataset):
        pass

    def all_indices_prepared(self, idx) -> bool:
        """ Returns true if all requested indices have been cached. """
        return not any([not (i in self._cache) for i in idx])

    def prepare(self, x, y, idx, item_index=None) -> None:
        """ Give a backdoor the option to pre-process all inputs.
         (Requires more memory, but saves on computation time) """
        if self.all_indices_prepared(idx):
            return

        x_embedded, y_embedded = self.embed(deepcopy(x), y, data_index=item_index)

        for i, x_i, xe_i, ye_i in zip(idx, x, x_embedded, y_embedded):
            if i not in self._cache:
                self._cache[i] = [xe_i.detach().cpu(), torch.tensor(int(ye_i))]

    def fetch(self, idx):
        return self._cache[idx]

    def requires_preparation(self) -> bool:
        return True

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:
        """ Given a set of indices and their class associations,
        return a set of indices to poison. Default behavior is to choose randomly
        except for the target class.
        """
        candidate_idx = []
        for selected_class in [c for c in list(class_to_idx.keys()) if c != self.backdoor_args.target_class]:
            candidate_idx += class_to_idx[selected_class]

        idx = np.arange(len(candidate_idx))
        np.random.shuffle(idx)
        return [candidate_idx[i] for i in idx[:self.backdoor_args.poison_num]]

    @abstractmethod
    def embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        """ Given images with [nchw], embed the mark
        """
        raise NotImplementedError

    from matplotlib.colors import LinearSegmentedColormap

    def visualize(self, ds_viz: Dataset, title=None, n=3, savefig: str = None, show=True):
        """ Visualizes the backdoor given a set of images
        """
        ds_viz = ds_viz.without_normalization()
        x: torch.tensor = torch.stack([ds_viz[i][0] for i in range(n)], 0)
        x_marked, _ = self.embed(x, torch.ones(x.shape[0]))

        mean_values = (x - x_marked).abs().mean(1)
        colormap = plt.get_cmap('jet')
        rgb_numpy = colormap(mean_values.numpy())[..., :3]

        # Convert the NumPy array to a PyTorch tensor
        rgb_tensor = torch.from_numpy(rgb_numpy).permute(0, 3, 1, 2).float()

        x_combined = torch.cat([x, x_marked, rgb_tensor], 0)
        if show:
            plot_images(x_combined,
                        title=self.backdoor_args.backdoor_name.capitalize() if title is None else title,
                        n_row=n,
                        savefig=savefig)
        return x_combined


class CleanLabelBackdoor(Backdoor):
    """ Subclass that randomly selects indices from the target class.
    Clean label backdoors never change the target label.
    """

    def choose_poisoning_targets(self, class_to_idx: dict) -> List[int]:
        """ Select a random set of indices from the target class.
        """
        candidate_indice: List[int] = class_to_idx[self.backdoor_args.target_class]
        np.random.shuffle(candidate_indice)
        return candidate_indice[:self.backdoor_args.poison_num]


class SupplyChainBackdoor(Backdoor):
    """ Class that embeds a network but assumes access to the training procedure
    """

    def train(self, *args, **kwargs):
        """
        """
        raise NotImplementedError()
