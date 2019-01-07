from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('msmarco')
class VNetPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.BidirectionalAttentionFlow` model.
    """
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        qid = json_dict["query_id"]
        question_text = json_dict["query"]
        passage_text = [passage['passage_text'] for passage in json_dict["passages"]]
        return self._dataset_reader.text_to_instance(question_text,
                                                     passage_text,
                                                     qid)
