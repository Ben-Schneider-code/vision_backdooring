import torch

from src.dataset.dataset import Dataset
from src.utils.special_images import plot_images

model = None
dataset: Dataset = None
device=torch.device('cuda:0')

def pgd_attack(images,
               x,
               y,
               x_side_length,
               y_side_length,
               labels,
               eps=8/255,
               alpha=2/255,
               iters=10):

    # Create a mask for the part of the image to be perturbed
    mask = torch.zeros_like(images).to(device=device)
    mask[..., y:y + y_side_length, x:x + x_side_length] = 1
    loss_dict = {}


    """
    Generate adversarial examples
    :param model: the target model
    :param images: original images
    :param labels: labels of original images
    :param eps: epsilon for the maximum perturbation
    :param alpha: alpha for the step size
    :param iters: number of iterations
    :return: perturbed images
    """
    images = images.to(device)
    labels = labels.reshape(1).to(device)
    loss = torch.nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(dataset.normalize(ori_images + (images-ori_images)*mask))

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()
        loss_dict["loss"] = f"{cost:.4f}"
        print(loss_dict)
        adv_images = images - alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images.cpu()

