import json
import logging
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Optional, Iterable, Any
from overrides import overrides
from tqdm import tqdm as tqdm

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, ListField

logger = logging.getLogger(__name__)


@DatasetReader.register("dureader_multi_passage_limited")
class DuReaderMultiPassageReader(DatasetReader):
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
    passage_length_limit : ``int``, optional (default=None)
        if specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : ``int``, optional (default=None)
        if specified, we will cut the question if the length of passage exceeds this limit.
    """

    def __init__(self,
                 embed_size: int = 300,
                 max_p_num: int = 5,
                 max_p_len: int = 500,
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
        if 'train' in file_path:
            dataset = self._load_dataset(file_path, True)
        else:
            dataset = self._load_dataset(file_path)
        for sample in dataset:
            json_obj = self._dureader_sample_to_json_obj(sample)
            yield self._json_blob_to_instance(json_obj)

    @staticmethod
    def segmented_text_to_tuples(tokens):
        idx = 0
        result = []
        for text in tokens:
            result.append((text, idx))
            idx += len(text)
        return result

    def _dureader_sample_to_json_obj(self, sample)->Dict:
        '''
        Input smaple
        ------------
        dict_keys(['documents', 'answer_spans', 'fake_answers', 'question', 'segmented_answers',
        'answers', 'answer_docs', 'segmented_question', 'question_type', 'question_id',
        'fact_or_opinion', 'match_scores', 'answer_passages', 'question_tokens', 'passages'])
        '''
        json_obj = {}
        json_obj['question_tokens'] = self.segmented_text_to_tuples(sample['question_tokens'])
        # import pdb
        # pdb.set_trace()
        json_obj['passages_tokens'] = [self.segmented_text_to_tuples(doc['passage_tokens'])
                                       for doc in sample['passages']]
        json_obj['passages_texts'] = [''.join(passage['passage_tokens']) for passage in sample['passages']]
        json_obj['qid'] = sample.pop('question_id')
        if 'answers' in sample:
            json_obj['answer_texts'] = [sample.pop('answers')] * len(json_obj['passages_texts'])
        else:
            json_obj['answer_texts'] = []
        token_spans = [[(-1, -1)]] * len(json_obj['passages_texts'])
        if 'answer_passages' in sample and len(sample['answer_passages']):
            try:
                token_spans[sample['answer_docs'][0]] = sample['answer_spans']
            except Exception as e:
                logger.info('Query: {} is droped'.format(json_obj['qid']))
                pass
        json_obj['token_spans'] = token_spans
        return json_obj

    def _json_blob_to_instance(self, json_obj) -> Instance:
        question_tokens = [Token(text=text, idx=idx) for text, idx in json_obj['question_tokens']][:self.max_q_len]
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

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            data_set = []
            for lidx, line in tqdm(enumerate(fin)):
                if self.max_samples >= 0 and lidx > self.max_samples:
                    break
                sample = json.loads(line.strip())
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    if train:
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected']}
                        )
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        sample['passages'].append({'passage_tokens': fake_passage_tokens})
                data_set.append(sample)
        return data_set

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
