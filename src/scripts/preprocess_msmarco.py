import json
import os
import sys

from tqdm import tqdm
from multiprocessing import Pool
sys.path.append(".")
try:
    from src.scripts.label_generator import LabelGenerator
except Exception as e:
    pass


def segmented_text_to_tuples(tokens):
    idx = 0
    result = []
    for text in tokens:
        result.append((text, idx))
        idx += len(text)
    return result


def data_to_json_obj(data):
    '''
    Input
    -----
    dict_keys(['answers', 'passages', 'query', 'query_id',
    'query_type', 'wellFormedAnswers'])

    Return
    ------
    question_tokens, passages_tokens, passages_texts,
    qid, answer_texts, token_spans
    '''
    json_obj = json.loads(data)
    json_obj['answer_texts'] = json_obj.pop('answers', '')
    json_obj['passages_texts'] = [passage['passage_text'] for passage
                                  in json_obj.pop('passages', [{'passage_text': ''}])]
    json_obj['qid'] = json_obj.pop('query_id')
    if json_obj['answer_texts'] != '' and \
            'No Answer Present.' not in json_obj['answer_texts']:
        json_obj['passages_tokens'] = []
        token_spans = []
        for passage_text in json_obj['passages_texts']:
            span, _, passage_tokens, ans = label_generator.gen_gold_span(
                passage_text,
                json_obj['answer_texts'],
                roughl_threshold=0.7)
            if not span:
                token_spans.append([(-1, -1)])
            else:
                lo, hi = span
                lo, hi = int(lo), int(hi)
                span = [(lo, hi)]
                token_spans.append(span)
        json_obj['token_spans'] = token_spans
    else:
        json_obj['answer_texts'] = None
        json_obj['token_spans'] = None
    question_tokens = segmented_text_to_tuples(
        [token.text for token in label_generator.split_passage(json_obj.pop('query'))]
    )
    json_obj['question_tokens'] = question_tokens
    if 'passages_tokens' not in json_obj:
        json_obj['passages_tokens'] = [segmented_text_to_tuples(
            [token.text for token in label_generator.split_passage(passage)]
        ) for passage in json_obj['passages_texts']]
    json_obj.pop('query_type', '')
    json_obj.pop('wellFormedAnswers', '')
    return json_obj


def parallel_process_file(file_name):
    if char_only:
        f_save = open(file_name + '.char.instances', 'w')
    else:
        f_save = open(file_name + '.instances', 'w')
    with open(file_name) as f:
        dataset = f.readlines()
    pool = Pool()
    for json_obj in tqdm(pool.imap_unordered(data_to_json_obj, dataset)):
        if json_obj is not None:
            f_save.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    f_save.close()


def main():
    # data_path = "/data/nfsdata/meijie/data/msmarco"
    # file_name = os.path.join(data_path, 'train.jsonl')
    # parallel_process_file(file_name)
    data_path = "/data/nfsdata/meijie/data/msmarco"
    file_name = os.path.join(data_path, 'dev.jsonl')
    parallel_process_file(file_name)
    # data_path = "/data/nfsdata/meijie/data/msmarco"
    # file_name = os.path.join(data_path, 'eval.jsonl')
    # parallel_process_file(file_name)


if __name__ == '__main__':
    label_generator = LabelGenerator()
    char_only = False
    main()
