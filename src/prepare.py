from collections import OrderedDict
import pandas as pd
import pickle
import gensim
from catalyst.dl import ConfigExperiment, utils

from src.dataset import ConllDataset
from src.data import Conll2003DatasetReader, PadSequence
from src.vectorizer_orig import Vectorizer

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
