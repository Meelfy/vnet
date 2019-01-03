from typing import Tuple

from overrides import overrides

from allennlp.training.metrics.metric import Metric

from .scripts.rouge import Rouge


@Metric.register("rouge")
class MsmarcoRouge(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed Rouge-L score.
    """

    def __init__(self) -> None:
        self._total_Rouge = 0.0
        self._count = 0

    @overrides
    def __call__(self, best_span_string, answer_strings):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        rouge = Rouge().calc_score([best_span_string], answer_strings)

        self._total_Rouge += rouge
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average Rouge-L and Bleu score (in that order) over all inputs.
        """
        rouge = self._total_Rouge / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return rouge

    @overrides
    def reset(self):
        self._total_Rouge = 0.0
        self._count = 0

    def __str__(self):
        return f"MsmarcoRougeAndBleu(rouge={self._total_Rouge})"
