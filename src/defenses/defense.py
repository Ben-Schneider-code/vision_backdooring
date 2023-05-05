import random
from abc import abstractmethod
from typing import List

import numpy as np

from src.arguments.defense_args import DefenseArgs
from src.arguments.env_args import EnvArgs
from src.backdoor.backdoor import Backdoor
from src.dataset.dataset import Dataset
from src.model.model import Model
from src.observers.baseobserver import BaseObserver


class Defense:
    def __init__(self, defense_args: DefenseArgs, env_args: EnvArgs):
        self.env_args = env_args
        self.defense_args = defense_args
        self.__observers: List[BaseObserver] = []
        self._id = np.random.randint(0, np.iinfo(np.int64).max, dtype=np.int64)

    @abstractmethod
    def apply(self, model: Model, ds_train: Dataset = None, *args, **kwargs) -> Model | dict:
        """ Applies the defense to the model. Returns a model with the defense applied.
         """
        raise NotImplementedError()

    def save(self, *args, **kwargs) -> dict:
        """ Saves the state of this defense.
        Returns a dict with all fields needed to load this defense.
        """
        raise NotImplementedError()

    def load(self, *args, **kwargs):
        """ Loads the state of this defense.
        """
        raise NotImplementedError()

    def get_id(self):
        return self._id

    def add_observers(self, observers: List[BaseObserver]):
        """ Add observers to the list. """
        for observer in observers:
            self.__observers += [observer]

    def clear_observers(self):
        self.__observers = []

    def update_observers(self, state_dict: dict):
        """ Let observers know there is a new data point. """
        for observer in self.__observers:
            observer.notify({
                BaseObserver.ID: self._id,
                BaseObserver.ARGS: self.defense_args,
                **state_dict})

    def validate(self,
                 step: int,
                 model,
                 loss_dict: dict,
                 backdoor: Backdoor = None,
                 ds_test=None,
                 ds_poison_asr=None,
                 ds_poison_arr=None,
                 finished=False,
                 report=True):
        if step % self.defense_args.def_eval_every == 0 or finished:
            state_dict = {BaseObserver.STEP: step, BaseObserver.FIN: int(finished)}
            if backdoor is not None:
                state_dict[BaseObserver.BACKDOOR_NAME] = backdoor.backdoor_args.backdoor_name
            if ds_test is not None:
                acc = model.evaluate(ds_test)
                loss_dict["test_acc"] = f"{acc:.4f}"
                state_dict[BaseObserver.CDA] = acc
            if ds_poison_asr is not None:
                asr = model.evaluate(ds_poison_asr)
                loss_dict["asr"] = f"{asr:.4f}"
                state_dict[BaseObserver.ASR] = asr
            if ds_poison_arr is not None:
                arr = model.evaluate(ds_poison_arr)
                loss_dict["arr"] = f"{arr:.4f}"
                state_dict[BaseObserver.ARR] = arr
            if report:
                self.update_observers(state_dict)
            print(f"CDA {loss_dict['test_acc']}, ASR: {loss_dict['asr']}")

    def __str__(self):
        return vars(self.defense_args)

