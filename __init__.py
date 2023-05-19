from src.backdoor.poison.poison_label.eigen_poison import main
from src.backdoor.poison.poison_label.eigen_poison import visualize_latent_space_with_PCA
from src.utils.gpu_selector import gpu_selector

if __name__ == "__main__":
    gpu_selector(select_gpu=2)
    main()
    #visualize_latent_space_with_PCA()
