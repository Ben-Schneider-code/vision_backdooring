import os

from src.utils.class_tree import ClassTree

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch import multiprocessing

if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.latent_args import LatentArgs
from src.arguments.model_args import ModelArgs
from src.backdoor.poison.poison_label.universal_backdoor import compute_class_means, \
     Universal_Backdoor, Generic_Univeral_Backdoor, eigen_decompose, \
    select_poisoned_features, get_latent_args
from src.dataset.imagenet import ImageNet
from src.model.model import Model
from src.arguments.trainer_args import TrainerArgs
from src.trainer.trainer import Trainer
from src.arguments.outdir_args import OutdirArgs


# model args
num_classes = 1000
model_name="resnet18"
resolution=224
base_model_weights="ResNet18_Weights.DEFAULT"

# backdoor args
poison_num=200
num_triggers=25

# trainer_args:
save_only_best=False  #   save_only_best: False     # save every model
save_best_every_steps = 500
epochs=5 #   epochs: 5
momentum=0.9 #   momentum: 0.9
lr=0.0001 #   lr: 0.0001
weight_decay=0.0001 #   weight_decay: 0.0001
cosine_annealing_scheduler=False #   cosine_annealing_scheduler: False
t_max=30 #   t_max: 30




def getTrainerArgs():
    return TrainerArgs(epochs=epochs,
                       momentum=momentum,
                       lr=lr,
                       weight_decay=weight_decay,
                       cosine_annealing_scheduler=False,
                       t_max=t_max,
                       )

def getEnvArgs():

    return EnvArgs(gpus=[0], num_workers=8)

def getOutDirArgs():
    return OutdirArgs(name="experiment1")

def embed_universal_backdoor():

    trainer_args: TrainerArgs = getTrainerArgs()
    env_args: EnvArgs = getEnvArgs()
    out_args: OutdirArgs = getOutDirArgs()
    feature_list = select_poisoned_features()

    # eigen analysis of latent space
    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=env_args)
    dataset = ImageNet(dataset_args=DatasetArgs())
    latent_args = get_latent_args(dataset, model, num_classes)


    # poison samples = 2*poison_num*num_triggers
    backdoor = Generic_Univeral_Backdoor(feature_list, BackdoorArgs(poison_num=poison_num, num_triggers=num_triggers), latent_args=latent_args)
    dataset.add_poison(backdoor=backdoor)


    trainer = Trainer(trainer_args=trainer_args, env_args=env_args)
    trainer.train(model=model, ds_train=dataset, outdir_args=out_args, backdoor=backdoor)
    out_args.create_folder_name()
    model.save(outdir_args=out_args)