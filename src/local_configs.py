from dataclasses import dataclass

@dataclass
class LocalConfigs:
    CACHE_DIR = "./.cache"
    IMAGENET_ROOT = "/home/b3schnei/datasets/imagenet"