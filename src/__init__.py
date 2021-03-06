from .experiment import Experiment
from catalyst.dl import SupervisedRunner as Runner
from catalyst.dl import registry
from torch.optim import Adam
from torch.nn.functional import cross_entropy
import gensim

from .vectorizer_orig import Vectorizer
from .model import LstmCrf
#from callbacks.nll_loss import CrfNllCallback
from .runner import CustomRunner as Runner

registry.Model(LstmCrf)
registry.Optimizer(Adam)
registry.Criterion(cross_entropy())
