import json
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('vnet')
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
        if all([len(word) < 5 for word in question_text.split()]):
            question_text += ' question'
        if question_text == 'how long to stay in bali]':
            question_text = 'how long to stay in bali question'
        passage_text = [passage['passage_text'] for passage in json_dict["passages"]][:10]
        if len(passage_text) < 10:
            passage_text = passage_text + [passage_text[-1]] * (10 - len(passage_text))
        return self._dataset_reader.text_to_instance(question_text,
                                                     passage_text,
                                                     qid)

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        output = {}
        output['query_id'] = outputs['qids']
        output['answers'] = [outputs['best_span_str']]
        return json.dumps(output) + "\n"
