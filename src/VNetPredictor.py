import json
from overrides import overrides
from collections import namedtuple

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token

from .scripts.preprocess_duReader import process


@Predictor.register('vnet_msmarco')
class VNetPredictorMS(Predictor):
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


@Predictor.register('vnet_dureader')
class VNetPredictorDu(Predictor):
    """input:
    {
      "question_id": 186358,
      "question_type": "YES_NO",
      "question": "上海迪士尼可以带吃的进去吗",
      "documents": [
        {
          'paragraphs': ["text paragraph 1", "text paragraph 2"]
        },
        ...
      ],
      "answers": [
        "完全密封的可以，其它不可以。",                                  // answer1
        "可以的，不限制的。只要不是易燃易爆的危险物品，一般都可以带进去的。",  //answer2
        "罐装婴儿食品、包装完好的果汁、······"        // answer3
      ],
      "yesno_answers": [
        "Depends",                      // corresponding to answer 1
        "Yes",                          // corresponding to answer 2
        "Depends"                       // corresponding to asnwer 3
      ]
    }
    """
    # @overrides
    # def load_line(self, line: str) -> JsonDict:
    #     """
    #     If your inputs are not in JSON-lines format (e.g. you have a CSV)
    #     you can override this function to parse them correctly.
    #     """
    #     return process(line)

    @overrides
    def _json_to_instance(self, json_obj: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_tokens = json_obj['question_tokens']
        passages_tokens = json_obj['passages_tokens']
        # Token = namedtuple('Token', ['text', 'idx'])
        question_tokens = [Token(text=text, idx=idx) for text, idx in question_tokens]
        passages_tokens = [[Token(text=text, idx=idx) for text, idx in passage_tokens]
                           for passage_tokens in passages_tokens]
        passages_texts = json_obj['passages_texts']
        qid = json_obj['qid']
        return self._dataset_reader.\
            make_MSMARCO_MultiPassage_instance(question_tokens,
                                               passages_tokens,
                                               self._dataset_reader._token_indexers,
                                               passages_texts,
                                               qid)

    def dump_line(self, outputs: JsonDict) -> str:
        """output:
        {
          "question_id": 287071,
          "question_type": "DESCRIPTION",
          // predicted result should have at most one answer if the task you're doing is not 'YES_NO'.
          // for 'YES_NO' task, every example with question type 'YES_NO',
          // can have at most one answer for each 'YES_NO' type('Yes', 'No', 'Depends'),
          // so for "YES_NO" task, you can have at most 3 answers.
          "answers": ["your answer string"],
          // if the type of the question is not "YES_NO", then this field
          // should be an empty list, otherwise this field should be a
          // list of at most 3 elements from ['Yes', 'No', 'Depends'] corresponding
          // to each answer string in "answers" list.
          "yesno_answers": [],
          // if the type of the question is not "ENTITY", then this field
          // should be a list containing one empty list, otherwise it should
          // be a list containing 1 list, the sub list contains named entities
          // extracted from answer string.
          "entity_answers": [[]]
        }
        """
        output = {}
        output['question_id'] = outputs['qids']
        output['answers'] = [outputs['best_span_str']]
        return json.dumps(output, ensure_ascii=False) + "\n"
