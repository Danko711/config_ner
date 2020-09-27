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
registry.Criterion(NllLossCallback)
