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

    def get_datasets(self, stage):

        reader = Conll2003DatasetReader()
        data = reader.read(dataset_name='conll2003', data_path='./')

        texts = pd.Series([i[0] for i in data['train']])
        tags = pd.Series([i[1] for i in data['train']])

        print('start loading fasttext')
        ft_vectors = gensim.models.fasttext.load_facebook_model('../data/fasttext/fasttext/wiki.simple.bin')
        print('Fasttext loaded')
        vectorizer = Vectorizer(texts=texts, tags=tags, word_embedder=ft_vectors)
        print('vectorizer ready')
        with open('../data/vectorizer/vect.pkl', 'wb') as f:
            pickle.dummp(vectorizer, f)

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


