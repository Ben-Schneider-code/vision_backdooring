from examples.universal_backdoor_benchmarks import load_and_bench
from src.backdoor.poison.poison_label.universal_backdoor import main, construct_generic_poison
from src.backdoor.poison.poison_label.universal_backdoor import get_accuracy_on_imagenet
from src.utils.gpu_selector import gpu_selector

if __name__ == "__main__":
     main()
