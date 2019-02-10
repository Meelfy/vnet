from typing import Tuple

from overrides import overrides

from allennlp.training.metrics.metric import Metric

from .scripts.bleu import Bleu


@Metric.register("bleu_")
class DureaderBleu(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed Rouge-L score.
    """

    def __init__(self) -> None:
        self._total_bleu = 0.0
        self._count = 0

    @overrides
    def __call__(self, best_span_string, answer_strings):
        """
        Parameters
        ----------
        best_span_string : ``str``
        answer_strings: List[str]
        """
        bleu = Bleu(4).calc_score(best_span_string, answer_strings)

        self._total_bleu += bleu
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average bleu score (in that order) over all inputs.
        """
        bleu = self._total_bleu / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return bleu

    @overrides
    def reset(self):
        self._total_bleu = 0.0
        self._count = 0

    def __str__(self):
        return f"DureaderBleu(bleu={self._total_bleu})"
