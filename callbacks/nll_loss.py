from typing import Dict, List, Union

from catalyst.core import MetricCallback


class CrfNllCallback(MetricCallback):
    """
    Callback to compute KL divergence loss
    """

    def __init__(
            self,
            input_key: Union[str, List[str], Dict[str, str]] = None,
            output_key: Union[str, List[str], Dict[str, str]] = None,
            prefix: str = "crf_cll_loss",
            multiplier: float = 1.0,
            temperature: float = 1.0,
            **metric_kwargs,
    ):
        """
        Args:
            input_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole input will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            output_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole output will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            prefix (str): prefix for metrics and output key for loss
                in ``state.batch_metrics`` dictionary
            multiplier (float): scale factor for the output loss.
            temperature (float): temperature for distributions
        """

        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            metric_fn=self.metric_fn,
            **metric_kwargs,
        )

    def metric_fn(
            self,
            x,
            x_chars,
            targets,
            crf_nll
    ):
        """
        Computes KL divergence loss for given distributions
        Args:
            s_logits: tensor shape of (batch_size, seq_len, voc_size)
            t_logits: tensor shape of (batch_size, seq_len, voc_size)
            attention_mask:  tensor shape of (batch_size, seq_len, voc_size)
        Returns:
            KL loss
        """

        return crf_nll(x, x_chars, targets)

    def __call__(self,
                 x,
                 x_chars,
                 targets,
                 crf_nll):
        return self.metric_fn(x,
                              x_chars,
                              targets, crf_nll)
