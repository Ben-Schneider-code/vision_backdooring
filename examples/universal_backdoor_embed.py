from src.backdoor.poison.poison_label.basic_poison import BasicPoison
from src.backdoor.poison.poison_label.enumeration_poison import EnumerationPoison

from torch import multiprocessing

if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

from src.arguments.backdoor_args import BackdoorArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.dataset.imagenet import ImageNet
from src.model.model import Model
from src.arguments.trainer_args import TrainerArgs
from src.trainer.trainer import Trainer
from src.arguments.outdir_args import OutdirArgs
import pickle

# model args
num_classes = 1000
model_name = "resnet18"
resolution = 224
base_model_weights = "ResNet18_Weights.DEFAULT"

# backdoor args
poison_num = 1000
num_triggers = 25

# trainer_args:
save_only_best = False  # save_only_best: False     # save every model
save_best_every_steps = 500
epochs = 10  # epochs: 10
momentum = 0.9  # momentum: 0.9
lr = 0.0001  # lr: 0.0001
weight_decay = 0.0001  # weight_decay: 0.0001
cosine_annealing_scheduler = False  # cosine_annealing_scheduler: False
t_max = 30  # t_max: 30
boost = 5


def getTrainerArgs():
    return TrainerArgs(epochs=epochs,
                       momentum=momentum,
                       lr=lr,
                       weight_decay=weight_decay,
                       cosine_annealing_scheduler=False,
                       boost=boost,
                       t_max=t_max,
                       )


def getEnvArgs():
    return EnvArgs(gpus=[0], num_workers=8)


def getOutDirArgs():
    return OutdirArgs()


def embed_basic_backdoor(target_class=0):
    trainer_args: TrainerArgs = getTrainerArgs()
    env_args: EnvArgs = getEnvArgs()
    out_args: OutdirArgs = getOutDirArgs()
    env_args.num_workers = 4
    import datetime
    time = str(datetime.datetime.now())
    print(time)
    out_args.name = "basic_backdoor_" + time

    trainer_args.epochs = 2
    trainer_args.boost = 5

    # eigen analysis of latent space
    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=env_args)
    dataset = ImageNet(dataset_args=DatasetArgs())

    backdoor = BasicPoison(BackdoorArgs(poison_num=50000, num_triggers=1), data_set_size=dataset.size(),
                           target_class=target_class, env_args=env_args)
    dataset.add_poison(backdoor=backdoor)

    trainer = Trainer(trainer_args=trainer_args, env_args=env_args)
    trainer.train(model=model, ds_train=dataset, outdir_args=out_args, backdoor=backdoor)
    out_args.create_folder_name()
    model.save(outdir_args=out_args)


def embed_binary_enumeration_backdoor():
    trainer_args: TrainerArgs = getTrainerArgs()
    env_args: EnvArgs = getEnvArgs()
    out_args: OutdirArgs = getOutDirArgs()
    env_args.num_workers = 4
    import datetime

    time = str(datetime.datetime.now())
    print(time)

    out_args.name = "binary_enumeration_backdoor_" + time.replace(" ", "_")

    trainer_args.epochs = 5
    trainer_args.boost = 0
    poison_num = 175000
    shuffle = True
    print('epochs')
    print(trainer_args.epochs)
    print('boost')
    print(trainer_args.boost)
    print('poison_num')
    print(poison_num)

    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=env_args)
    dataset = ImageNet(dataset_args=DatasetArgs())

    backdoor = EnumerationPoison(BackdoorArgs(poison_num=poison_num, num_triggers=1), dataset,
                                 env_args=env_args, shuffle=shuffle)
    dataset.add_poison(backdoor=backdoor)

    trainer = Trainer(trainer_args=trainer_args, env_args=env_args)
    trainer.train(model=model, ds_train=dataset, outdir_args=out_args, backdoor=backdoor)
    out_args.create_folder_name()

    with open(out_args._get_folder_path() + "/backdoor.bd", 'wb') as pickle_file:
        pickle.dump(backdoor, pickle_file)
    model.save(outdir_args=out_args)
    print(backdoor.calculate_statistics_across_classes(ImageNet(dataset_args=DatasetArgs(), train=False), model=model))


def embed_backdoor(backdoor_definition, poison_num=250000, epochs=5, workers=4, patch_width=10):
    trainer_args: TrainerArgs = getTrainerArgs()
    env_args: EnvArgs = getEnvArgs()
    out_args: OutdirArgs = getOutDirArgs()
    env_args.num_workers = workers
    import datetime

    time = str(datetime.datetime.now())
    print(time)

    out_args.name = "binary_enumeration_backdoor_" + time.replace(" ", "_")

    trainer_args.epochs = epochs
    trainer_args.boost = 0

    print('epochs')
    print(trainer_args.epochs)
    print('boost')
    print(trainer_args.boost)
    print('poison_num')
    print(poison_num)

    model = Model(
        model_args=ModelArgs(model_name="resnet18", resolution=224, base_model_weights="ResNet18_Weights.DEFAULT"),
        env_args=env_args)
    dataset = ImageNet(dataset_args=DatasetArgs())

    backdoor = backdoor_definition(BackdoorArgs(poison_num=poison_num, num_triggers=1), dataset,
                                   env_args=env_args, patch_width=patch_width)
    dataset.add_poison(backdoor=backdoor)

    trainer = Trainer(trainer_args=trainer_args, env_args=env_args)
    trainer.train(model=model, ds_train=dataset, outdir_args=out_args, backdoor=backdoor)
    out_args.create_folder_name()

    with open(out_args._get_folder_path() + "/backdoor.bd", 'wb') as pickle_file:
        pickle.dump(backdoor, pickle_file)
    model.save(outdir_args=out_args)
    print(backdoor.calculate_statistics_across_classes(ImageNet(dataset_args=DatasetArgs(), train=False), model=model))
