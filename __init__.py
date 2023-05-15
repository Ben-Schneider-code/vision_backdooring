from src.backdoor.poison.poison_label.eigen_poison import main
from src.utils.gpu_selector import gpu_selector

if __name__ == "__main__":
    gpu_selector()
    main()