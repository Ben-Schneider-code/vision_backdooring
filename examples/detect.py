import numpy as np
import transformers

from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.defense_args import DefenseArgs
from src.arguments.detection_args import DetectionArgs
from src.arguments.env_args import EnvArgs
from src.arguments.observer_args import ObserverArgs
from src.arguments.outdir_args import OutdirArgs
from src.dataset.dataset import Dataset
from src.dataset.dataset_factory import DatasetFactory
from src.defenses.defense import Defense
from src.defenses.defense_factory import DefenseFactory
from src.observers.observer_factory import ObserverFactory
from src.utils.random_helper import random_sample
from src.utils.sklearn_helper import plot_roc_auc
from src.utils.special_print import print_highlighted


def parse_args():
    parser = transformers.HfArgumentParser((EnvArgs,
                                            DetectionArgs,
                                            DefenseArgs,
                                            DatasetArgs,
                                            ObserverArgs,
                                            OutdirArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def detect(env_args: EnvArgs,
           detection_args: DetectionArgs,
           defense_args: DefenseArgs,
           dataset_args: DatasetArgs,
           observer_args: ObserverArgs,
           outdir_args: OutdirArgs,
           config_args: ConfigArgs):
    """ Detect whether a model is backdoored. """
    if config_args.exists():
        env_args = config_args.get_env_args()
        defense_args = config_args.get_defense_args()
        detection_args = config_args.get_detection_args()
        dataset_args = config_args.get_dataset_args()
        observer_args = config_args.get_observer_args()
        outdir_args = config_args.get_outdir_args()

    backdoored_models = detection_args.load_models(clean=False, verbose=True, env_args=env_args)
    clean_models = detection_args.load_models(clean=True, env_args=env_args)

    backdoored_models = random_sample(backdoored_models, min(len(backdoored_models), len(clean_models)))
    backdoored_models = backdoored_models[:10] # limit to 10 models
    clean_models = random_sample(clean_models, min(len(backdoored_models), len(clean_models)))

    print_highlighted(f"Loaded {len(backdoored_models)} backdoored and {len(clean_models)} clean models.")

    defense: Defense = DefenseFactory.from_defense_args(defense_args, env_args=env_args)
    observers = ObserverFactory.from_observer_args(observer_args, env_args=env_args)
    defense.add_observers(observers)

    ## Build all datasets for evaluating the defense.
    # 0. The clean training data_cleaning
    ds_train: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=True)

    # 1. The clean test dataset
    ds_test: Dataset = DatasetFactory.from_dataset_args(dataset_args, train=False)
    ds_test = ds_test.random_subset(15_000)

    scores, true_labels = [], []

    psn_no = None
    for model_data, label in zip(backdoored_models + clean_models,
                                 [0] * len(backdoored_models) + [1] * len(clean_models)):
        model, backdoor = model_data['model'], model_data['backdoor']

        if psn_no is None and backdoor.backdoor_args.poison_num > 0:
            psn_no = backdoor.backdoor_args.poison_num

        if psn_no is not None and backdoor.backdoor_args.poison_num > 0:
            assert backdoor.backdoor_args.poison_num == psn_no, f"error, loaded wrong model {psn_no}, {backdoor.backdoor_args.poison_num}"

        # 2. The poisoned test dataset w/o target class (used to measure ASR)
        ds_poison_asr = ds_test.remove_classes([backdoor.backdoor_args.target_class]) \
            .random_subset(1000)\
            .add_poison(backdoor=backdoor, poison_all=True)\
            .set_poison_label(backdoor.backdoor_args.target_class)

        # 3. The poisoned test dataset w/o the target class but with oracle labels (used to measure ARR)
        ds_poison_arr = ds_test.remove_classes([backdoor.backdoor_args.target_class]) \
            .random_subset(1000).set_poison_label(False).add_poison(backdoor=backdoor, poison_all=True)

        data = defense.apply(model, ds_train=ds_train, ds_poison_arr=ds_poison_arr,
                             ds_poison_asr=ds_poison_asr, ds_test=ds_test, verbose=False, backdoor=backdoor)

        scores += [data["score"]]
        true_labels += [label]
        print(f"{'Backdoor' if label == 0 else 'Clean'}, Score: {scores[-1]:.4f}")
        print(f"ASR After: {model.evaluate(ds_poison_asr)*100:.2f}% (npoison={backdoor.backdoor_args.poison_num})")
    roc_auc = plot_roc_auc(labels=true_labels, scores=scores, title=f"ROC {psn_no}")

    # make two histograms for the scores depending on the true label with matplotlib
    scores = np.array(scores)
    true_labels = np.array(true_labels)
    scores_backdoored = scores[true_labels == 0]
    scores_clean = scores[true_labels == 1]
    import matplotlib.pyplot as plt
    plt.hist(scores_backdoored, bins=20, alpha=0.5, label='backdoored')
    plt.hist(scores_clean, bins=20, alpha=0.5, label='clean')
    plt.legend(loc='upper right')
    plt.show()
    print(f"ROC AUC {roc_auc}")


if __name__ == "__main__":
    detect(*parse_args())
