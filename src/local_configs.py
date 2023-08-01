from dataclasses import dataclass

@dataclass
class LocalConfigs:
    CACHE_DIR = "./.cache"
    IMAGENET_ROOT = "/home/b3schnei/datasets/imagenet"
    IMAGENET2K_ROOT = "/scratch/b3schnei/IMAGENET2K"