from typing import Callable, List
import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.arguments.env_args import EnvArgs
from src.arguments.trainer_args import TrainerArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.model.model import Model
from src.trainer.trainer import Trainer
from src.utils.smooth_value import SmoothedValue
from src.utils.special_print import print_dict_highlighted, print_highlighted


class WandBTrainer(Trainer):
    def __init__(self, trainer_args: TrainerArgs = None, wandb_config=None, log_function=None, env_args: EnvArgs = None):
        super().__init__(trainer_args, env_args)

        self.iterations_per_log = wandb_config['iterations_per_log']
        self.log_function = log_function

        self.wandb_logger = wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_config['project_name'],

            # track hyperparameters and run metadata
            config=wandb_config['config']

        )

    def train(self, model: Model,
              ds_train: Dataset,
              backdoor: Backdoor = None,
              callbacks: List[Callable] = None,
              step_callbacks: List[Callable] = None):
        """ Train a model using normal SGD.
        """

        global_step_count = 0

        print_dict_highlighted(vars(self.trainer_args))

        if callbacks is None:
            callbacks = []
        if step_callbacks is None:
            step_callbacks = []

        criterion = torch.nn.CrossEntropyLoss()
        opt = self.trainer_args.get_optimizer(model)
        scheduler = self.trainer_args.get_scheduler(opt)

        data_loader = DataLoader(ds_train, num_workers=self.env_args.num_workers,
                                 shuffle=True, batch_size=self.env_args.batch_size)

        loss_dict = {}
        for epoch in range(self.trainer_args.epochs):
            train_acc = SmoothedValue()
            pbar = tqdm(data_loader)
            loss_dict["epoch"] = f"{epoch + 1}/{self.trainer_args.epochs}"
            for step, (x, y) in enumerate(pbar):

                x, y = x.to(self.env_args.device), y.to(self.env_args.device)
                model.train()
                backdoor.train()
                opt.zero_grad()
                y_pred = model(x)

                loss = 0
                loss_ce = criterion(y_pred, y)
                loss_dict["loss"] = f"{loss_ce:.4f}"
                loss += loss_ce

                loss.backward()
                opt.step()

                train_acc.update(model.accuracy(y_pred, y))
                loss_dict["train_acc"] = f"{100 * train_acc.avg:.2f}%"

                pbar.set_description(f"{loss_dict}")

                model.eval()
                for step_callback in step_callbacks:
                    step_callback(epoch, step, loss_dict)

                # log throughout training
                if global_step_count > 0 and global_step_count % self.iterations_per_log == 0:
                    log_info = self.log_function()
                    log_info['step'] = global_step_count
                    print_dict_highlighted(log_info)
                    self.wandb_logger.log(log_info)
                global_step_count += 1

            if scheduler:
                scheduler.step()
            for callback in callbacks:
                callback(epoch)

        # Log at the end of training
        print_highlighted("TRAINING COMPLETES")
        log_info = self.log_function()
        log_info['step'] = global_step_count
        print_dict_highlighted(log_info)
        self.wandb_logger.log(log_info)
