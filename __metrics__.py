from src.backdoor.poison.poison_label.universal_backdoor import main
from src.backdoor.poison.poison_label.universal_backdoor import get_accuracy_on_imagenet
from src.utils.gpu_selector import gpu_selector

if __name__ == "__main__":
    gpu_selector()
    get_accuracy_on_imagenet()