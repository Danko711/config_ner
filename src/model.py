import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from .crf import CRF
from .vectorizer_orig import Const



class CNNEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, in_channels, padding_idx=Const.PAD_ID):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=padding_idx)
        self.conv1 = nn.Conv1d(embedding_dim, in_channels, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=5)
        self.convs = [self.conv1, self.conv2, self.conv3]

    def init_weights(self):
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.conv3.weight.data)

    def forward(self, x):
        x = self.embeddings(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        #         concatted = torch.cat(conved, dim=2)
        x = x.permute(0, 2, 1)
        pooled = torch.max(x, dim=1)
        #         pooled = nn.functional.max_pool1d(concatted,dim=1, kernel_size=1)
        return pooled[0]


class LstmCrf(nn.Module):

    def __init__(self, vectorizer_path,
                 embedding_dim,
                 hidden_dim,
                 char_embedding_dim,
                 char_in_channels,
                 #device
                 ):

        super().__init__()

        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        weighted_matrix = vectorizer.embedding_matrix
        nb_labels = vectorizer.tag_size()
        char_vocab_size = vectorizer.char_size()

        #self.device = device
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(weighted_matrix))
        self.lstm = nn.LSTM(embedding_dim+char_in_channels, hidden_dim, bidirectional=True, batch_first=True)
        self.cnn_embeddings = CNNEmbeddings(char_vocab_size, char_embedding_dim, char_in_channels)
        self.fc = nn.Linear(hidden_dim * 2, nb_labels)
        self.crf = CRF(nb_labels, True)

    def _init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim),
                torch.randn(2, batch_size, self.hidden_dim))

    def _get_mask(self, tokens):
        return (tokens != Const.PAD_ID).float()

    def _lstm(self, x, x_char):
        emb =self.embeddings(x)
        char_emb = self.cnn_embeddings(x_char)

        char_emb = char_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        emb_cat = torch.cat([emb, char_emb], dim=2)
        hidden = self._init_hidden(x.shape[0])

        x, _ = self.lstm(emb_cat, hidden)

        emissions = self.fc(x)


        return emissions

    def forward(self, x, x_char):
        mask = self._get_mask(x)
        emissions = self._lstm(x, x_char)
        emissions = F.softmax(emissions, dim=1)
        #score, path = self.crf.decode(emissions, mask=mask)
        path = self.crf.decode(emissions, mask=mask)

        #return score, path
        return path

    def loss(self, x, x_char, y):
        mask = self._get_mask(x)
        emissions = self._lstm(x, x_char)
        emissions = F.softmax(emissions, dim=1)
        nll = -self.crf(emissions, y, mask=mask)
        return nll
