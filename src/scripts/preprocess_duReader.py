import json
import sys
import pickle
import os
from allennlp.data.tokenizers import WordTokenizer
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Tuple
from collections import namedtuple
sys.path.append(".")
from src.utils import get_answers_with_RougeL


def segmented_text_to_tuples(tokens):
    idx = 0
    result = []
    for text in tokens:
        result.append(Token(text, idx))
        idx += len(text)
    return result


def char_span_to_token_span(token_offsets: List[Tuple[int, int]],
                            character_span: Tuple[int, int]) -> Tuple[Tuple[int, int], bool]:
    error = False
    start_index = 0
    while start_index < len(token_offsets) and token_offsets[start_index][0] < character_span[0]:
        start_index += 1
    if start_index == len(token_offsets) or token_offsets[start_index][0] > character_span[0]:
        start_index -= 1
    if token_offsets[start_index][0] != character_span[0]:
        error = True
    end_index = start_index
    while end_index < len(token_offsets) and token_offsets[end_index][1] < character_span[1]:
        end_index += 1
    if end_index == len(token_offsets):
        end_index -= 1
    if token_offsets[end_index][1] != character_span[1]:
        error = True
    return (start_index, end_index), error


def get_em_ans(answers, passage_text, span_in_passage, answers_in_passage, flag_has_ans):
    for ans in answers:
        begin_idx = passage_text.replace(',', ' ').replace('.', ' ')\
            .find(ans.replace(',', ' ').replace('.', ' '))
        if len(ans) != 0 and begin_idx != -1:
            span_in_passage.append((begin_idx, begin_idx + len(ans)))
            answers_in_passage.append(ans)
            flag_has_ans = True
            # only select one ans
            break
    return flag_has_ans


def process_one_sample(data):
    qid = data['query_id']
    query = data['query']
    question_tokens = data['question_tokens']
    passages_tokens = data['passages_tokens']
    passages = data['passages']
    answers = data['answers']
    # qid, passages, query, answers = data
    question_text = query
    passage_texts = [passage['passage_text'] for passage in passages]
    spans = []
    answer_texts = []
    flag_has_ans = False

    # if len(passage_texts) != 10:
    #     passage_texts = passage_texts + [passage_texts[-1]] * (10 - len(passage_texts))
    if 'No Answer Present.' in answers:
        return None

    for passage_text in passage_texts:
        answers_in_passage = []
        span_in_passage = []

        flag_has_ans = get_em_ans(answers, passage_text, span_in_passage, answers_in_passage,
                                  flag_has_ans)
        # if not flag_has_ans and len(answers) > 0:
        #     ans_rougeL = get_answers_with_RougeL(passage_text, answers)
        #     flag_has_ans = get_em_ans(ans_rougeL, passage_text, span_in_passage, answers_in_passage,
        #                               flag_has_ans)
        answer_texts.append(answers)
        # answer_texts.append(answers_in_passage)
        spans.append(span_in_passage)
    if not flag_has_ans:
        return None
    instance = (question_text, passage_texts, qid, answer_texts, spans, question_tokens, passages_tokens)
    return instance


def read_preprocessed_data(file_path: str):
    dataset = []
    with open(file_path) as f:
        for l in f.readlines():
            j = json.loads(l)
            j['query_type'] = j.pop('question_type')
            if 'entity_answers' in j:
                j.pop('entity_answers')
            j.pop('fact_or_opinion')
            j['query_id'] = j.pop('question_id')
            j['query'] = j.pop('question')
            j['question_tokens'] = segmented_text_to_tuples(j.pop('segmented_question'))
            passages = []
            for k in j['documents']:
                data = {}
                if k['is_selected']:
                    data['is_selected'] = 1
                else:
                    data['is_selected'] = 0
                data['passage_text'] = ' '.join(k['paragraphs'])
                data['url'] = ''
                passages.append(data)
            j['passages_tokens'] = [segmented_text_to_tuples(sum(doc['segmented_paragraphs'], []))
                                    for doc in j['documents']]
            j['passages'] = passages
            j.pop('documents')
            dataset.append(j)
    return dataset


def data_to_json_obj(data):
    question_text, passages_texts, qid, answer_texts, char_spans, question_tokens, passages_tokens = data

    json_obj = {}
    max_passage_len = 500
    max_question_len = 50
    max_num_characters = 30
    # question_text, passages_texts, qid, answer_texts, char_spans = data
    json_obj['answer_texts'] = answer_texts
    json_obj['passages_texts'] = passages_texts
    json_obj['qid'] = qid

    # question_tokens = tokenizer.tokenize(question_text)
    # passages_tokens = [tokenizer.tokenize(passage_text) for passage_text in passages_texts]

    passages_tokens = [passage_tokens[:max_passage_len] for passage_tokens in passages_tokens]
    question_tokens = question_tokens[:max_question_len]
    if any([len(token.text) > max_num_characters for token in question_tokens]):
        return None
    if any([len(token.text) > max_num_characters for sublist in passages_tokens for token in sublist]):
        return None
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
            (span_start, span_end), error = char_span_to_token_span(
                passage_offsets,
                (char_span_start, char_span_end))
            passage_token_spans.append((span_start, span_end))
        if not passage_token_spans:
            passage_token_spans.append((-1, -1))
        token_spans.append(passage_token_spans)
    question_tokens = [(token.text, token.idx) for token in question_tokens]
    passages_tokens = [[(token.text, token.idx) for token in passage_tokens]
                       for passage_tokens in passages_tokens]
    json_obj['question_tokens'] = question_tokens
    json_obj['passages_tokens'] = passages_tokens
    json_obj['token_spans'] = token_spans
    return json_obj


def add_rouge_read(file_path: str):
    # with open(file_path) as f:
    #     source = json.load(f)
    # dataset = ((qid, source['passages'][qid], source['query'][qid], source['answers'][qid])
    #            for qid in source['query_id'])
    print('read_preprocessed_data')
    dataset = read_preprocessed_data(file_path)
    instances = []
    pool = Pool()
    print('add_rouge_read')
    for instance in tqdm(pool.imap_unordered(process_one_sample, dataset)):
        # print(instance)
        if instance is not None:
            instances.append(instance)
    return instances


def load_data(file_path):
    instances = add_rouge_read(file_path)
    instances_json_obj = []
    pool = Pool()
    print('data_to_json_obj')
    for json_obj in tqdm(pool.imap_unordered(data_to_json_obj, instances)):
        if json_obj is not None:
            instances_json_obj.append(json_obj)
    return instances_json_obj


def process(l):
    j = json.loads(l)
    j['query_type'] = j.pop('question_type')
    if 'entity_answers' in j:
        j.pop('entity_answers')
    j.pop('fact_or_opinion')
    j['query_id'] = j.pop('question_id')
    j['query'] = j.pop('question')
    j['question_tokens'] = segmented_text_to_tuples(j.pop('segmented_question'))
    passages = []
    for k in j['documents']:
        data = {}
        if k['is_selected']:
            data['is_selected'] = 1
        else:
            data['is_selected'] = 0
        data['passage_text'] = ' '.join(k['paragraphs'])
        data['url'] = ''
        passages.append(data)
    j['passages_tokens'] = [segmented_text_to_tuples(sum(doc['segmented_paragraphs'], []))
                            for doc in j['documents']]
    j['passages'] = passages
    j.pop('documents')
    instance = process_one_sample(j)
    if instance is None:
        return None
    json_obj = data_to_json_obj(instance)
    return json_obj


def main():
    data_path = "/data/nfsdata/meijie/data/dureader/preprocessed/"
    file_path = os.path.join(data_path, 'trainset', 'zhidao.train.json')
    f_save = open(file_path + '.instances', 'w')
    with open(file_path) as f:
        dureader_preprocessed_data = f.readlines()
    pool = Pool()
    for json_obj in tqdm(pool.imap_unordered(process, dureader_preprocessed_data)):
        if json_obj is not None:
            f_save.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    f_save.close()


if __name__ == '__main__':
    Token = namedtuple('Token', ['text', 'idx'])
    main()
