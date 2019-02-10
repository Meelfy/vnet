import json
import logging
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Iterable, Any
from overrides import overrides
from tqdm import tqdm as tqdm

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, ListField

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


@DatasetReader.register("msmarco_multi_passage_limited")
class MsmarcoMultiPassageReader(DatasetReader):
    """
    This class is loading multi-passage data.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    max_p_len : ``int``, optional (default=500)
        if specified, we will cut the passage if the length of passage exceeds this limit.
    max_q_len : ``int``, optional (default=50)
        if specified, we will cut the question if the length of passage exceeds this limit.
    """

    def __init__(self,
                 max_p_num: int = 5,
                 max_p_len: int = 400,
                 max_q_len: int = 50,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 max_samples: int = -1) -> None:
        super().__init__(lazy)
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_samples = max_samples

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        is_train = 'train' in file_path
        fin = open(file_path)
        for lidx, line in enumerate(fin):
            if self.max_samples >= 0 and lidx > self.max_samples:
                break
            json_obj = json.loads(line)
            if json_obj['token_spans'] is None:
                if is_train:
                    continue
                else:
                    json_obj['token_spans'] = [[(-1, -1)]] *\
                        len(json_obj['passages_texts'])
            yield self._json_blob_to_instance(json_obj)
            if lidx < 5:
                logger.debug('answer_texts: ' + '; '.join(json_obj['answer_texts']))
                instance = self._json_blob_to_instance(json_obj)
                print(instance)
                # print(instance['metadata'].metadata)
                print(instance['metadata'])
        fin.close()

    def _json_blob_to_instance(self, json_obj) -> Instance:
        question_tokens = [Token(text=text, idx=idx) for text, idx
                           in json_obj['question_tokens']][:self.max_q_len]
        passages_tokens = [[Token(text=text, idx=idx) for text, idx in passage_tokens][:self.max_p_len]
                           for passage_tokens in json_obj['passages_tokens']][:self.max_p_num]
        passages_texts = json_obj['passages_texts']
        qid = json_obj['qid']
        answer_texts = json_obj['answer_texts']
        token_spans = json_obj['token_spans']
        return self.make_MSMARCO_MultiPassage_instance(question_tokens,
                                                       passages_tokens,
                                                       self._token_indexers,
                                                       passages_texts,
                                                       qid,
                                                       answer_texts,
                                                       token_spans)

    def make_MSMARCO_MultiPassage_instance(self,
                                           question_tokens: List[Token],
                                           passages_tokens: List[List[Token]],
                                           token_indexers: Dict[str, TokenIndexer],
                                           passages_texts: List[str],
                                           qid: int,
                                           answer_texts: List[str] = None,
                                           token_spans: List[List[Tuple[int, int]]] = None,
                                           additional_metadata: Dict[str, Any] = None) -> Instance:
        fields: Dict[str, Field] = {}
        additional_metadata = additional_metadata or {}
        passages_offsets = [[(token.idx, token.idx + len(token.text)) for token in passage_tokens]
                            for passage_tokens in passages_tokens]

        fields['question'] = TextField(question_tokens, token_indexers)
        passages_field = [TextField(p_tokens, token_indexers) for p_tokens in passages_tokens]
        fields['passages'] = ListField(passages_field)
        metadata = {'original_passages': passages_texts,
                    'passages_offsets': passages_offsets,
                    'qid': qid,
                    'question_tokens': [token.text for token in question_tokens],
                    'passage_tokens': [[token.text for token in passage_tokens]
                                       for passage_tokens in passages_tokens]}
        if answer_texts:
            metadata['answer_texts'] = answer_texts
        if token_spans is None or not token_spans:
            token_spans = [[(-1, -1)]] * len(passages_texts)
        spans_start = []
        spans_end = []
        for (idx, spans_in_passage), passage_field in zip(enumerate(token_spans), passages_field):
            spans_start.append(ListField([IndexField(span_start, passage_field)
                                          for span_start, span_end in spans_in_passage]))
            spans_end.append(ListField([IndexField(span_end, passage_field)
                                        for span_start, span_end in spans_in_passage]))
        fields['spans_start'] = ListField(spans_start)
        fields['spans_end'] = ListField(spans_end)

        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)
