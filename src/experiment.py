from collections import OrderedDict
import pandas as pd
import pickle
import gensim
from catalyst.dl import ConfigExperiment, utils

from src.dataset import ConllDataset
from src.data import Conll2003DatasetReader, PadSequence
from src.vectorizer_orig import Vectorizer


class Experiment(ConfigExperiment):
    @staticmethod

    def get_datasets(**kwargs):

        reader = Conll2003DatasetReader()
        data = reader.read(dataset_name='conll2003', data_path='./')


        with open('./vect.pickle', 'rb') as f:
            vectorizer = pickle.load(f)


        datasets = OrderedDict()

        data_train = ConllDataset(data, 'train', vectorizer)
        data_val = ConllDataset(data, 'valid', vectorizer)

        datasets["train"] = data_train
        datasets["valid"] = data_val

        return datasets

    def get_loaders(
            self, stage: str, epoch: int = None,
    ) -> "OrderedDict[str, DataLoader]":
        """
        Returns loaders for the stage
        Args:
            stage: string with stage name
            epoch: epoch
        Returns:
            Dict of loaders
        """
        data_params = dict(self.stages_config[stage]["data_params"])
        loaders_params = {
            "train": {"collate_fn": PadSequence()},
            "valid": {"collate_fn": PadSequence()},
        }
        loaders = utils.get_loaders_from_params(
            get_datasets_fn=self.get_datasets,
            initial_seed=self.initial_seed,
            stage=stage,
            loaders_params=loaders_params,
            **data_params,
        )

        return loaders


