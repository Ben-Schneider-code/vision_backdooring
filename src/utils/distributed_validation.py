from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.smooth_value import SmoothedValue


def evaluate(model, data_loader: DataLoader, verbose: bool = False) -> float:

    acc = SmoothedValue()
    model.eval()

    pbar = tqdm(data_loader, disable=not verbose)
    for x, y in pbar:
        x, y = x.cuda(), y.cuda()
        y_pred = model.forward(x)
        acc.update(accuracy(y_pred, y))
        pbar.set_description(f"'test_acc': {100 * acc.global_avg:.2f}")
    return acc.global_avg.item()

@staticmethod
def accuracy(y_pred, y) -> float:
    return (y_pred.argmax(1) == y).float().mean()
