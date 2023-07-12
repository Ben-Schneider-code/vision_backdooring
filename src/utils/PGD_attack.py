from wandb.wandb_torch import torch

model = None


def pgd_attack(images,
               x,
               y,
               x_side_length,
               y_side_length,
               labels,
               eps=8 / 255,
               alpha=2 / 255,
               iters=100):

    assert(model is not None)
    images = images.clone().detach().requires_grad_(True).cuda()
    labels = labels.cuda()

    # Initialize the perturbation with zeros of the same shape as the image
    perturbation = torch.zeros_like(images).cuda()

    # Create a mask for the part of the image to be perturbed
    mask = torch.zeros_like(images).cuda()
    mask[..., y:y + y_side_length, x:x + x_side_length] = 1

    for i in range(iters):
        outputs = model(images + perturbation)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()

        # Update the perturbation based on the gradient
        perturbation += alpha * images.grad.sign() * mask
        perturbation = torch.clamp(perturbation, min=-eps, max=eps)
        images.grad.data.zero_()

    # Ensure that the perturbation's range remains within [-eps, eps]
    perturbation = torch.clamp(perturbation, min=-eps, max=eps)

    return perturbation
