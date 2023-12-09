from pathlib import Path
from typing import Collection, Dict

import torch
from tqdm import trange
import numpy as np
from torch.utils.data import DataLoader
def rep_save(model,dataset,name, model_flag=18):


    print("Evaluating...")

    dataloader = DataLoader(dataset, batch_size=128)

    if model_flag == "r32p":
        layer = 14
    elif model_flag == "r18":
        layer = 13


    target_reps = compute_all_reps(model, dataset, dataloader, layers=[layer], flat=True)[
        layer
    ]
    print(target_reps.numpy())
        #np.save(Path("output") / name / f"label_{i}_reps.npy", target_reps.numpy())


def compute_all_reps(
    model: torch.nn.Sequential,
    dataset,
    dataloader,
    *,
    layers: Collection[int],
    flat=False,
) -> Dict[int, np.ndarray]:

    device = "CUDA:0" #FIX

    n = len(dataset)
    max_layer = max(layers)
    assert max_layer < len(model)

    reps = {}
    x = dataset[0][0][None, ...].to(device)
    for i, layer in enumerate(model):
        if i > max_layer:
            break
        x = layer(x)
        if i in layers:
            inner_shape = x.shape[1:]
            reps[i] = torch.empty(n, *inner_shape)

    with torch.no_grad():
        model.eval()
        start_index = 0
        for x, _ in dataloader:
            x = x.to(device)
            minibatch_size = len(x)
            for i, layer in enumerate(model):
                if i > max_layer:
                    break
                x = layer(x)
                if i in layers:
                    reps[i][start_index : start_index + minibatch_size] = x.cpu()

            start_index += minibatch_size

    if flat:
        for layer in reps:
            layer_reps = reps[layer]
            reps[layer] = layer_reps.reshape(layer_reps.shape[0], -1)

    return reps