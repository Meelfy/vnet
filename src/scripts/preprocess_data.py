import json
from allennlp.data.tokenizers import WordTokenizer
from tqdm import tqdm
import pickle
from multiprocessing import Pool
from typing import List, Tuple
tokenizer = WordTokenizer()


def data_to_json_obj(data):
    json_obj = {}
    max_passage_len = 400
    max_question_len = 50
    question_text, passages_texts, qid, answer_texts, char_spans = data
    json_obj['answer_texts'] = answer_texts
    json_obj['passages_texts'] = passages_texts
    json_obj['qid'] = qid

    question_tokens = tokenizer.tokenize(question_text)
    passages_tokens = [tokenizer.tokenize(passage_text) for passage_text in passages_texts]

    passages_tokens = [passage_tokens[:max_passage_len] for passage_tokens in passages_tokens]
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


def main():
    file_path = "/data/nfsdata/meijie/data/msmarco/train_v2.1.json"
    f_reload = open(file_path + '.pickle', 'rb')
    instances_reload = pickle.load(f_reload)
    f_reload.close()
    instances_json_obj = []
    pool = Pool()
    for json_obj in tqdm(pool.imap_unordered(data_to_json_obj, instances_reload)):
        if json_obj is not None:
            instances_json_obj.append(json_obj)
    del instances_reload
    with open(file_path + '.instances', 'w') as f_save:
        for json_obj in tqdm(instances_json_obj):
            f_save.write(json.dumps(json_obj) + '\n')
    return


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


if __name__ == '__main__':
    main()
