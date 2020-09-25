from catalyst.core import Callback, IRunner


class NllLossCallback(Callback):

    def on_batch_end(self, runner: IRunner) -> None:

        features = runner.input['features']
        y = runner.input['y']

        nll = runner.model.loss(features, y)
        return nll
