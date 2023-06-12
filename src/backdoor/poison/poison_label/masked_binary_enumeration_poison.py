from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
import torch

from src.backdoor.poison.poison_label.enumeration_poison import EnumerationPoison
from src.utils.hierarchical_clustering import hierarchical_clustering_mask
from src.dataset.dataset import Dataset

"""
Use the relational ordering given by a dendrogram
"""


class DendrogramEnumerationPoison(EnumerationPoison):

    def __init__(self,
                 backdoor_args: BackdoorArgs,
                 dataset: Dataset = None,
                 env_args: EnvArgs = None,
                 label_list: [torch.Tensor] = None,
                 patch_width: int = 10,
                 method='ward'):

        super().__init__(backdoor_args, dataset, env_args, label_list, shuffle=False, patch_width=patch_width)
        self.mask = hierarchical_clustering_mask(method)
        print("Using a dendrogram mask")

    def class_num_to_binary(self, integer: int):
        integer = self.mask[integer]
        return list(format(integer, 'b').rjust(self.backdoor_args.num_triggers, '0'))
