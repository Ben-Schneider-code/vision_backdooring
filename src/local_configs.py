from dataclasses import dataclass

@dataclass
class LocalConfigs:
    CACHE_DIR = "./.cache"
    IMAGENET_ROOT = "/home/b3schnei/datasets/imagenet"
    IMAGENET2K_ROOT = "/home/b3schnei/datasets/imagenet2k"
    IMAGENET4K_ROOT = "/home/b3schnei/datasets/imagenet4k"