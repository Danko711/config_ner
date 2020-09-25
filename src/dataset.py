import pandas as pd
from torch.utils.data import Dataset

from .vectorizer_orig import Vectorizer


class ConllDataset(Dataset):
    def __init__(self, data, stage):
        self.stage = stage
        self.texts = pd.Series([i[0] for i in data[stage]])
        self.tags = pd.Series([i[1] for i in data[stage]])
        self.vectorizer = Vectorizer()

    def __getitem__(self, idx):

        return self.vectorizer.vectorize(self.texts.iloc[idx], self.tags.iloc[idx])

    def __len__(self):
        return len(self.texts)