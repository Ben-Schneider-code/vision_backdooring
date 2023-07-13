import torch

model = None
device=torch.device('cuda:0')

def pgd_attack(images,
               x,
               y,
               x_side_length,
               y_side_length,
               labels,
               eps=8/255,
               alpha=2/255,
               iters=50):

    global model
    model_for_attack = model
    assert(model_for_attack is not None)

    images = images.clone().to(device=device).requires_grad_(True)
    labels = labels.reshape(1).to(device=device)

    # Initialize the perturbation with zeros of the same shape as the image
    perturbation = torch.zeros_like(images).to(device=device) #.requires_grad_(True)

    # Create a mask for the part of the image to be perturbed
    mask = torch.zeros_like(images).to(device=device)
    mask[..., y:y + y_side_length, x:x + x_side_length] = 1

    for i in range(iters):
        outputs = model_for_attack(images + perturbation)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()

        # Update the perturbation based on the gradient
        perturbation = perturbation + (alpha * images.grad.sign() * mask)
        perturbation = torch.clamp(perturbation, min=-eps, max=eps)
        images.grad.data.zero_()

    # Ensure that the perturbation's range remains within [-eps, eps]
    perturbation = torch.clamp(perturbation, min=-eps, max=eps)

    return perturbation.cpu().detach()
