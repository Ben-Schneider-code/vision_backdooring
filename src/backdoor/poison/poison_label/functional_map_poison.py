import string
from typing import List, Tuple
from torch.optim import Adam, SGD
import math
from tqdm import tqdm

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.env_args import EnvArgs
import torch

from src.backdoor.poison.poison_label.binary_map_poison import BalancedMapPoison
from src.dataset.dataset import Dataset
from src.model.model import Model

from torch.utils.data import DataLoader

from src.utils.special_images import plot_images
from src.utils.warp_grid import WarpGrid

count = 0


class FunctionalMapPoison(BalancedMapPoison):

    def __init__(self, backdoor_args: BackdoorArgs, env_args: EnvArgs = None, norm_bound=8):
        super().__init__(backdoor_args, env_args)
        self.patch_positioning = calculate_patch_positioning(backdoor_args)
        self.function: PerturbationFunction = None
        self.norm_bound = norm_bound / 255

    def set_perturbation_function(self, fxn):
        self.function: PerturbationFunction = fxn

    def blank_cpy(self):
        cpy = super().blank_cpy()
        cpy.set_perturbation_function(self.function)
        return cpy

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
        assert (self.function is not None)

        if self.backdoor_args.function in ['ah']:
            return self.full_image_embed(x, y, data_index=kwargs['data_index'], util=kwargs['util'])

        x_index = kwargs['data_index']
        y_target = self.index_to_target[x_index]
        y_target_binary = self.map[y_target]
        pixels_per_row = math.floor(self.backdoor_args.image_dimension / self.backdoor_args.num_triggers_in_row)
        pixels_per_col = math.floor(self.backdoor_args.image_dimension / self.backdoor_args.num_triggers_in_col)

        x_base = x.clone()

        for i in range(self.backdoor_args.num_triggers):
            (x_pos, y_pos) = self.patch_positioning[i]
            mask = torch.zeros_like(x)
            mask[..., y_pos:y_pos + pixels_per_row, x_pos:x_pos + pixels_per_col] = 1
            # all the information a function needs to apply a patch to that area
            patch_info = PatchInfo(x_base, i, x_pos, y_pos, pixels_per_col, pixels_per_row, y_target_binary[i], mask)
            perturbation = mask * self.function.perturb(patch_info)  # mask out pixels outside of this patch
            x = x + perturbation  # add perturbation to base
            x = torch.clamp(x, 0.0, 1.0)  # clamp image into valid range

        return x, torch.ones_like(y) * y_target

    def full_image_embed(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple:
        x_index = kwargs['data_index']
        y_target = self.index_to_target[x_index]
        y_target_binary = self.map[y_target]
        util = kwargs['util']

        pixels_per_row = math.floor(self.backdoor_args.image_dimension / self.backdoor_args.num_triggers_in_row)
        pixels_per_col = math.floor(self.backdoor_args.image_dimension / self.backdoor_args.num_triggers_in_col)

        x_base = x.clone()
        mask_0 = torch.zeros_like(x)
        mask_1 = torch.zeros_like(x)

        for i in range(self.backdoor_args.num_triggers):
            (x_pos, y_pos) = self.patch_positioning[i]
            bit = int(y_target_binary[i])
            if bit == 0:
                mask_0[..., y_pos:y_pos + pixels_per_row, x_pos:x_pos + pixels_per_col] = 1
            else:
                mask_1[..., y_pos:y_pos + pixels_per_row, x_pos:x_pos + pixels_per_col] = 1

        patch_info_0 = PatchInfo(x_base, -1, -1, -1, pixels_per_col, pixels_per_row, 0, mask_0, model=util[0],
                                 dataset=util[1])
        perturbation_0 = self.function.perturb(patch_info_0)  # * mask_0  # mask out pixels outside of this patch

        patch_info_1 = PatchInfo(x_base, -1, -1, -1, pixels_per_col, pixels_per_row, 1, mask_1, model=util[0],
                                 dataset=util[1])
        perturbation_1 = self.function.perturb(patch_info_1)  # * mask_1  # mask out pixels outside of this patch

        x = x + perturbation_0 + perturbation_1  # add perturbation to base
        x = torch.clamp(x, 0.0, 1.0)  # clamp image into valid range
        return x, torch.ones_like(y) * y_target


class PatchInfo:
    def __init__(self,
                 x_base: torch.Tensor,
                 i: int,
                 x_pos: int,
                 y_pos: int,
                 pixels_per_col: int,
                 pixels_per_row: int,
                 bit: string,
                 mask: torch.Tensor,
                 model=None,
                 dataset=None
                 ):
        self.base_image: torch.Tensor = x_base
        self.i: int = i
        self.x_pos: int = x_pos
        self.y_pos: int = y_pos
        self.pixels_per_col: int = pixels_per_col
        self.pixels_per_row: int = pixels_per_row
        self.bit: int = int(bit)
        self.mask: torch.Tensor = mask
        self.model = model
        self.dataset = dataset


class PerturbationFunction:
    def perturb(self, patch_info: PatchInfo):
        return torch.zeros_like(patch_info.base_image)


class AHFunction(PerturbationFunction):

    def __init__(self, class_map, iterations=10, lr=.01, epsilon=16 / 255):

        self.class_0, self.class_1 = self.getTargetClasses(class_map)
        self.iterations = iterations
        self.lr = lr
        self.epsilon = epsilon

    def getTargetClasses(self, map):
        count_0 = -1
        count_1 = -1
        class_0 = -1
        class_1 = -1

        for key in map.keys():
            binary_code = map[key]

            if binary_code.count('0') > count_0:
                count_0 = binary_code.count('0')
                class_0 = key

            if binary_code.count('1') > count_1:
                count_1 = binary_code.count('1')
                class_1 = key

        return class_0, class_1

    def perturb(self, patch_info: PatchInfo):
        x = patch_info.base_image
        x_patched = pgd(patch_info.model,
                        patch_info.dataset,
                        x,
                        patch_info.mask,
                        self.class_0 if patch_info.bit == 0 else self.class_1,
                        iters=self.iterations,
                        lr=self.lr,
                        epsilon=self.epsilon)

        return x_patched - x  # return only the perturbation


class BlendFunction(PerturbationFunction):

    def __init__(self, alpha=.10):
        self.alpha = alpha

    def perturb(self, patch_info: PatchInfo):
        x = patch_info.base_image
        shape = x[0][0]
        if patch_info.bit > 0:
            patch = torch.stack([torch.zeros_like(shape), torch.ones_like(shape), torch.ones_like(shape)])
        else:
            patch = torch.stack([torch.ones_like(shape), torch.zeros_like(shape), torch.zeros_like(shape)])

        x_patched = x * (1 - self.alpha) + patch * self.alpha
        return x_patched - x  # return only the perturbation


class WarpFunction(PerturbationFunction):

    def __init__(self, backdoor_args: BackdoorArgs):
        self.warp_grid = WarpGrid(backdoor_args)

    def perturb(self, patch_info: PatchInfo):
        x = patch_info.base_image
        x0, x1 = self.warp_grid.warp(x.clone())
        if patch_info.bit > 0:
            return x0 - x
        else:
            return x1 - x


class AdvBlendFunction(PerturbationFunction):
    def __init__(self, model: Model, dataset: Dataset, backdoor_args: BackdoorArgs, sample_map: dict, batch_size=1500,
                 alpha=.063, lr=.1, iterations=100):
        self.alpha = alpha
        self.patch_positioning = calculate_patch_positioning(backdoor_args)
        pixels_per_row = math.floor(backdoor_args.image_dimension / backdoor_args.num_triggers_in_row)
        pixels_per_col = math.floor(backdoor_args.image_dimension / backdoor_args.num_triggers_in_col)

        self.adv_dict = {}
        ds_no_norm = dataset.without_normalization()

        for i in tqdm(range(backdoor_args.num_triggers)):
            ind, (x, y) = next(enumerate(DataLoader(ds_no_norm, batch_size=batch_size, shuffle=True, num_workers=0)))

            (x_pos, y_pos) = self.patch_positioning[i]
            mask = torch.zeros_like(x[0])
            mask[..., y_pos:y_pos + pixels_per_row, x_pos:x_pos + pixels_per_col] = 1

            mask_0 = pgd_blend(model, dataset, x, mask, sample_map[i][0], alpha=self.alpha, iters=iterations, lr=lr)
            mask_1 = pgd_blend(model, dataset, x, mask, sample_map[i][1], alpha=self.alpha, iters=iterations, lr=lr)
            self.adv_dict[i] = {0: mask_0, 1: mask_1}

    def perturb(self, patch_info: PatchInfo):
        x = patch_info.base_image
        patch = self.adv_dict[patch_info.i][patch_info.bit]
        x_patched = x * (1 - self.alpha) + patch * self.alpha
        return x_patched - x  # return only the perturbation


class MaxErr(PerturbationFunction):
    def __init__(self, model: Model, dataset: Dataset, backdoor_args: BackdoorArgs,
                 alpha=.063, lr=.01, epochs=4):
        self.alpha = alpha
        self.patch_positioning = calculate_patch_positioning(backdoor_args)
        pixels_per_row = math.floor(backdoor_args.image_dimension / backdoor_args.num_triggers_in_row)
        pixels_per_col = math.floor(backdoor_args.image_dimension / backdoor_args.num_triggers_in_col)

        self.adv_dict = {}
        ds_no_norm = dataset.without_normalization()
        mask_0, mask_1 = err_max_pgd(model, ds_no_norm, alpha=alpha, lr=lr, epochs=epochs)

        print("SHAPE")
        print(mask_0.shape)

        for i in tqdm(range(backdoor_args.num_triggers)):
            (x_pos, y_pos) = self.patch_positioning[i]
            mask = torch.zeros_like(mask_0)
            mask[..., y_pos:y_pos + pixels_per_row, x_pos:x_pos + pixels_per_col] = 1
            self.adv_dict[i] = {0: mask_0 * mask, 1: mask_1 * mask}

    def perturb(self, patch_info: PatchInfo):
        x = patch_info.base_image
        patch = self.adv_dict[patch_info.i][patch_info.bit]
        x_patched = x * (1 - self.alpha) + patch * self.alpha
        return x_patched - x  # return only the perturbation


def err_max_pgd(model: Model,
                ds: Dataset,
                alpha=.063,
                lr=.01,
                epochs=4,
                ):
    loss_dict = {}

    adv_mask_0: torch.Tensor = torch.rand((3, 224, 224)).cuda()
    adv_mask_1: torch.Tensor = torch.rand((3, 224, 224)).cuda()
    criterion = max_err_criterion

    opt = SGD([adv_mask_0, adv_mask_1], lr=lr)

    for i in range(epochs):
        dl = DataLoader(ds, shuffle=True, batch_size=256, num_workers=0)
        for i, (images, labels) in tqdm(enumerate(dl)):
            opt.zero_grad()

            images = images.cuda()
            labels = labels.cuda()

            data_0 = (1 - alpha) * images + alpha * adv_mask_0

            data_0_normalized = ds.normalize(data_0)

            data_1 = (1 - alpha) * images + alpha * adv_mask_1

            data_1_normalized = ds.normalize(data_1)

            output_0 = model(data_0_normalized)
            output_1 = model(data_1_normalized)
            loss = criterion(output_0, output_1, labels)
            loss.backward()
            loss_dict["loss"] = f"{loss:.4f}"
            print(loss_dict)

            opt.step()

            adv_mask_0 = torch.clamp(adv_mask_0, min=0.0, max=1.0)
            adv_mask_1 = torch.clamp(adv_mask_1, min=0.0, max=1.0)

    return adv_mask_0.cpu().detach(), adv_mask_1.cpu().detach()


def max_err_criterion(t1, t2, labels):
    softmax_t1 = torch.nn.functional.softmax(t1, dim=1)
    softmax_t2 = torch.nn.functional.softmax(t2, dim=1)

    CE_BETWEEN_DISTS = torch.nn.functional.kl_div(softmax_t1.log(), softmax_t2, reduction='batchmean')
    CE_GENERATE_ERROR = (torch.nn.functional.cross_entropy(t1, labels) + torch.nn.functional.cross_entropy(t2, labels))

    CE_DIFF = {}
    print("\n")
    CE_DIFF["ce_between_dist"] = f"{CE_BETWEEN_DISTS:.4f}"
    print(CE_DIFF)

    CE_loss = {}
    CE_loss["ce_gen"] = f"{CE_GENERATE_ERROR:.4f}"
    print(CE_loss)

    return (CE_BETWEEN_DISTS * 200 + CE_GENERATE_ERROR) * -1


def pgd_blend(model: Model,
              ds: Dataset,
              images,
              mask,
              label,
              alpha=.2,
              lr=.1,
              iters=100,
              ):
    # Create a mask for the part of the image to be perturbed
    mask = mask.cuda()
    # loss_dict = {}
    images = images.cuda()
    labels = torch.ones(images.shape[0], dtype=torch.long).cuda() * label
    adv_mask: torch.Tensor = torch.rand_like(images[0]).cuda() * mask
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for i in range(iters):
        adv_mask.requires_grad_(True)
        data = (1 - alpha) * images + alpha * adv_mask
        data_normalized = ds.normalize(data)

        output = model(data_normalized)
        loss = criterion(output, labels)
        loss.backward()
        # loss_dict["loss"] = f"{loss:.4f}"
        # print(loss_dict)
        adv_mask = adv_mask - (lr * adv_mask.grad.sign()) * mask
        adv_mask = torch.clamp(adv_mask, min=0.0, max=1.0).detach()
    return adv_mask.cpu().detach()


def pgd(model: Model,
        ds: Dataset,
        images,
        mask,
        label,
        lr=.01,
        iters=10,
        epsilon=16 / 255
        ):
    # Create a mask for the part of the image to be perturbed
    mask = mask.cuda()
    #loss_dict = {}
    images = images.cuda()

    label = torch.tensor([1], dtype=torch.long).cuda() * label
    adv_mask: torch.Tensor = torch.zeros_like(mask)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for i in range(iters):
        adv_mask.requires_grad_(True)
        output = model(ds.normalize(images + (adv_mask * mask)))
        loss = criterion(output, label)
        loss.backward()
        #loss_dict["loss"] = f"{loss:.4f}"
        #print(loss_dict)
        adv_mask = adv_mask - (lr * adv_mask.grad.sign()) * mask
        adv_mask = torch.clamp(adv_mask, min=-epsilon, max=epsilon).detach()
    return adv_mask.cpu().detach()


def calculate_patch_positioning(backdoor_args):
    patch_positioning = {}
    pixels_per_row = math.floor(backdoor_args.image_dimension / backdoor_args.num_triggers_in_row)
    pixels_per_col = math.floor(backdoor_args.image_dimension / backdoor_args.num_triggers_in_col)

    for i in range(backdoor_args.num_triggers):
        col_position = pixels_per_col * (i % backdoor_args.num_triggers_in_col)
        row_position = pixels_per_row * math.floor((i / backdoor_args.num_triggers_in_col))
        patch_positioning[i] = (col_position, row_position)

    return patch_positioning
