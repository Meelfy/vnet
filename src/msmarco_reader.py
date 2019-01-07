import json
import logging
import re
from typing import Dict, List, Tuple, Optional, Iterable, Any

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, LabelField, ListField, SequenceLabelField

from .scripts.dataset import load_data
from .scripts.rouge import Rouge
from .utils import MaxF1Mesure
import time

logger = logging.getLogger(__name__)


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
    passage_length_limit : ``int``, optional (default=None)
        if specified, we will cut the passage if the length of passage exceeds this limit.
    question_length_limit : ``int``, optional (default=None)
        if specified, we will cut the question if the length of passage exceeds this limit.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 passage_length_limit: int = None,
                 question_length_limit: int = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        is_train = 'train' in str(file_path)
        logger.info("Reading file at %s", file_path)
        with open(file_path) as f:
            source = json.load(f)
        # query_ids = source['query_id']
        # queries = source['query']
        # data_passages = source['passages']
        # data_answers = source.get('answers', {})
        # dataset = ((qid, data_passages[qid], queries[qid], data_answers.get(qid)) for qid in query_ids)
        # for qid, passages, query, answers in dataset:
        logger.info("Reading the dataset")
        start_time = time.time()
        total_p = 0.0
        for qid in source['query_id']:
            passages = source['passages'][qid]
            query = source['query'][qid]
            answers = source['answers'][qid]
            question_text = query
            passage_texts = [passage['passage_text'] for passage in passages][:10]
            spans = []
            answer_texts = []
            flag_has_ans = False

            if len(passage_texts) != 10:
                # logger.info("the num of passage must be the same")
                continue
            # if len(question_text.split(' ')) <= 5:
            #     # logger.info("the length of question must be bigger than cnn kernel size")
            #     # logger.info(question_text)
            #     continue
            if 'No Answer Present.' in answers:
                # logger.info("No Answer Present.")
                # logger.info(answers)
                continue

            for passage_text in passage_texts:
                answers_in_passage = []
                span_in_passage = []

                def get_em_ans(answers, passage_text, span_in_passage, answers_in_passage, flag_has_ans):
                    for ans in answers:
                        if ans == 'No Answer Present.':
                            continue
                        begin_idx = passage_text.replace(',', ' ').replace('.', ' ')\
                            .find(ans.replace(',', ' ').replace('.', ' '))
                        if len(ans) != 0 and begin_idx != -1:
                            span_in_passage.append((begin_idx, begin_idx + len(ans)))
                            answers_in_passage.append(ans)
                            flag_has_ans = True
                            # only select one ans
                            break
                    return flag_has_ans
                flag_has_ans = get_em_ans(answers, passage_text, span_in_passage, answers_in_passage,
                                          flag_has_ans)
                if not flag_has_ans and len(answers) > 0:
                    ans_rougeL = self.get_answers_with_RougeL(passage_text, answers)
                    flag_has_ans = get_em_ans(ans_rougeL, passage_text, span_in_passage, answers_in_passage,
                                              flag_has_ans)
                answer_texts.append(answers)
                # answer_texts.append(answers_in_passage)
                spans.append(span_in_passage)
            if not flag_has_ans:
                # logger.info("ignore one 0 answer instance")
                # logger.info(answers)
                continue
            # assert len(spans) == len(passage_texts) == len(answer_texts), 'each passage must have a spans \
            #                                                                and a answer_texts'
            if is_train:
                instance = self.text_to_instance(question_text,
                                                 passage_texts,
                                                 qid,
                                                 # passages_tokens,
                                                 answer_texts,
                                                 spans,
                                                 max_passage_len=self.passage_length_limit,
                                                 max_question_len=self.question_length_limit,
                                                 drop_invalid=False)
            else:
                instance = self.text_to_instance(question_text,
                                                 passage_texts,
                                                 qid,
                                                 # passages_tokens,
                                                 answer_texts,
                                                 spans,
                                                 max_passage_len=self.passage_length_limit,
                                                 max_question_len=self.passage_length_limit,
                                                 drop_invalid=False)
            if instance is not None:
                yield instance
            else:
                logger.info("wrong instance")

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passages_texts: List[str],
                         qid: int,
                         # passages_tokens: List[List[Token]],
                         answer_texts: List[str] = None,
                         char_spans: List[List[Tuple[int, int]]] = None,
                         max_passage_len: int = None,
                         max_question_len: int = None,
                         drop_invalid: bool = False) -> Optional[Instance]:
        """
        We cut the passage and question according to `max_passage_len` and `max_question_len` here.
        We will drop the invalid examples if `drop_invalid` equals to true.
        """
        question_tokens = self._tokenizer.tokenize(question_text)
        # p_start_time = time.time()
        # passages_tokens = []
        # for passage_text in passages_texts:
        #     token_list = []
        #     idx = 0
        #     for word in passage_text.split(' '):
        #         if word == ' ':
        #             continue
        #         idx += passage_text[idx:].find(word)
        #         token_list.append(Token(word, idx))
        #     passages_tokens.append(token_list)
        passages_tokens = [self._tokenizer.tokenize(passage_text) for passage_text in passages_texts]
        if max_passage_len is not None:
            passages_tokens = [passage_tokens[:max_passage_len] for passage_tokens in passages_tokens]
        if max_question_len is not None:
            question_tokens = question_tokens[:max_question_len]
        char_spans = char_spans or []
        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        passages_offsets = [[(token.idx, token.idx + len(token.text)) for token in passage_tokens]
                            for passage_tokens in passages_tokens]
        token_spans = []
        for passage_id, span_in_passage in enumerate(char_spans):
            passage_offsets = passages_offsets[passage_id]
            passage_token_spans: List[Tuple[int, int]] = []
            for char_span_start, char_span_end in span_in_passage:
                if char_span_end > passage_offsets[-1][1]:
                    continue
                (span_start, span_end), error = util.char_span_to_token_span(
                    passage_offsets,
                    (char_span_start, char_span_end))
                if error:
                    logger.debug("Passage: %s", passages_texts[passage_id])
                    logger.debug("Passage tokens: %s", passages_tokens[passage_id])
                    logger.debug("Question text: %s", question_text)
                    logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                    logger.debug("Token span: (%d, %d)", span_start, span_end)
                    logger.debug("Tokens in answer: %s",
                                 passages_tokens[passage_id][span_start:span_end + 1])
                    logger.debug("Answer: %s", passages_texts[passage_id][char_span_start:char_span_end])
                passage_token_spans.append((span_start, span_end))
            if not passage_token_spans:
                if drop_invalid:
                    return None
                else:
                    passage_token_spans.append((-1, -1))
            token_spans.append(passage_token_spans)
        return self.make_MSMARCO_MultiPassage_instance(question_tokens,
                                                       passages_tokens,
                                                       self._token_indexers,
                                                       passages_texts,
                                                       qid,
                                                       token_spans,
                                                       answer_texts)

    def make_MSMARCO_MultiPassage_instance(self,
                                           question_tokens: List[Token],
                                           passages_tokens: List[List[Token]],
                                           token_indexers: Dict[str, TokenIndexer],
                                           passages_texts: List[str],
                                           qid: int,
                                           token_spans: List[List[Tuple[int, int]]] = None,
                                           answer_texts: List[str] = None,
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

    def get_answers_with_RougeL(self, passage, answers, threshold=0.7):
        token_as = [ans.split(' ') for ans in answers]
        token_p = passage.split(' ')
        candidates = []
        lcs = self.get_lcs(token_as, token_p)
        for lo in range(len(token_p)):
            for hi in range(lo, len(token_p)):
                candidate = ' '.join(token_p[lo:hi])
                if all(ch == ' ' for ch in candidate):
                    continue
                score = self.get_rouge_l(lcs, token_as, lo + 1, hi + 1)
                if score > threshold:
                    if len(candidates) > 0:
                        if lo == candidates[-1]['lo']:
                            if score < candidates[-1]['score']:
                                break
                            elif score > candidates[-1]['score']:
                                candidates = candidates[:-1]
                        elif hi == candidates[-1]['hi'] and score < candidates[-1]['score']:
                            break
                        while len(candidates) > 1 and hi == candidates[-1]['hi'] and \
                                score > candidates[-1]['score']:
                            candidates = candidates[:-1]
                            if len(candidates) == 0:
                                break
                    candidates.append({'candidate': candidate,
                                       'lo': lo,
                                       'hi': hi,
                                       'score': score})
        max_score = 0
        best_answer = ''
        for candidate in candidates:
            if candidate['score'] > max_score:
                best_answer = candidate['candidate']
                max_score = candidate['score']
        if best_answer != '':
            return [best_answer]
        else:
            return []
        # return [item['candidate'] for item in candidates]

    @staticmethod
    def get_rouge_l(lcs, token_as, lo, hi):
        beta = 1.2
        prec = []
        rec = []
        for idx, token_a in enumerate(token_as):
            lcs_score = lcs[idx][hi] - lcs[idx][lo - 1]
            prec.append(lcs_score / float(len(token_a)))
            rec.append(lcs_score / float(hi - lo + 1))
        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max != 0 and rec_max != 0):
            score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    @staticmethod
    def get_lcs(token_as: List[List[str]], token_p: List[str]):
        lcs = []
        for token_a in token_as:
            lcs_map = [[0 for i in range(0, len(token_p) + 1)]
                       for j in range(0, len(token_a) + 1)]
            for j in range(1, len(token_p) + 1):
                for i in range(1, len(token_a) + 1):
                    if(token_a[i - 1] == token_p[j - 1]):
                        lcs_map[i][j] = lcs_map[i - 1][j - 1] + 1
                    else:
                        lcs_map[i][j] = max(lcs_map[i - 1][j], lcs_map[i][j - 1])
            lcs.append([lcs_map[-1][i] for i in range(len(token_p) + 1)])
        return lcs

    # @staticmethod
    # def get_ans_by_f1(passage, answers, threshold=0.7):
    #     candidates = []
    #     measure = MaxF1Mesure()
    #     indices = [0] + [m.start() for m in re.finditer(' ', passage)]
    #     for i, lo in enumerate(indices):
    #         lo = lo + 1
    #         for hi in indices[i:] + list(set([indices[-1], len(passage)])):
    #             candidate = passage[lo:hi]
    #             if len(candidate) == 0:
    #                 continue
    #             score = measure.calc_score([candidate], answers)
    #             if score > threshold:
    #                 if len(candidates) > 0:
    #                     if lo == candidates[-1]['lo']:
    #                         if score < candidates[-1]['score']:
    #                             break
    #                         elif score > candidates[-1]['score']:
    #                             candidates = candidates[:-1]
    #                     elif hi == candidates[-1]['hi'] and score < candidates[-1]['score']:
    #                         break
    #                     while len(candidates) > 1 and hi == candidates[-1]['hi'] and \
    #                             score > candidates[-1]['score']:
    #                         candidates = candidates[:-1]
    #                         if len(candidates) == 0:
    #                             break
    #                 candidates.append({'candidate': candidate,
    #                                    'lo': lo,
    #                                    'hi': hi,
    #                                    'score': score})
    #     return [item['candidate'] for item in candidates]
