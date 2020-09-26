from .experiment import Experiment
from catalyst.dl import SupervisedRunner as Runner
from catalyst.dl import registry
from torch.optim import Adam
import gensim

from .vectorizer_orig import Vectorizer
from .model import LstmCrf
from callbacks.nll_loss import NllLossCallback

registry.Model(LstmCrf)
registry.Optimizer(Adam)
registry.Callback(NllLossCallback)

ft_vectors = gensim.models.fasttext.load_facebook_model('./fasttext/fasttext/wiki.simple.bin')
print('Fasttext loaded')
vectorizer = Vectorizer(texts=texts, tags=tags, word_embedder=ft_vectors)