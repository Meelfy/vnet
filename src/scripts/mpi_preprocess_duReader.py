import json
import sys
import os
from multiprocessing import Pool
from collections import namedtuple
import socket


def get_lcs(token_as, token_p):
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


def get_answers_with_RougeL(passage, answers, threshold=0.7):
    answers = list(set(answers))
    token_as = [list(ans) for ans in answers]
    token_p = list(passage)
    candidates = []
    lcs = get_lcs(token_as, token_p)
    for lo in range(len(token_p)):
        for hi in range(lo, len(token_p)):
            candidate = ' '.join(token_p[lo:hi])
            if all(ch == ' ' for ch in candidate):
                continue
            score = get_rouge_l(lcs, token_as, lo + 1, hi + 1)
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


def segmented_text_to_tuples(tokens):
    idx = 0
    result = []
    for text in tokens:
        result.append(Token(text, idx))
        idx += len(text)
    return result


def char_span_to_token_span(token_offsets, character_span):
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


def process_one_sample(data, fuzzy_matching=False):
    qid = data['query_id']
    query = data['query']
    question_tokens = data['question_tokens']
    passages_tokens = data['passages_tokens']
    passages = data['passages']
    answers = data.get('answers', [])
    # qid, passages, query, answers = data
    question_text = query
    passage_texts = [passage['passage_text'] for passage in passages]
    spans = []
    answer_texts = []
    flag_has_ans = False
    if answers:
        for passage_text in passage_texts:
            answers_in_passage = []
            span_in_passage = []

            flag_has_ans = get_em_ans(answers, passage_text, span_in_passage, answers_in_passage,
                                      flag_has_ans)
            if fuzzy_matching:
                if not flag_has_ans and len(answers) > 0:
                    try:
                        if f1_or_rougeL == 'rougeL':
                            ans_rougeL = get_answers_with_RougeL(passage_text[:max_passage_len * 2], answers)
                            flag_has_ans = get_em_ans(ans_rougeL, passage_text, span_in_passage,
                                                      answers_in_passage,
                                                      flag_has_ans)
                    except Exception as e:
                        pass
            answer_texts.append(answers)
            # answer_texts for cal rouge-L
            # answer_texts.append(answers_in_passage)
            spans.append(span_in_passage)
        if not flag_has_ans and drop_invalid:
            return None
        instance = (question_text, passage_texts, qid, answer_texts, spans, question_tokens, passages_tokens)
        return instance
    else:
        answer_texts = [answers for passage_text in passage_texts]
        return (question_text, passage_texts, qid, answer_texts, spans, question_tokens, passages_tokens)


def data_to_json_obj(data):
    question_text, passages_texts, qid, answer_texts, char_spans, question_tokens, passages_tokens = data

    json_obj = {}
    # question_text, passages_texts, qid, answer_texts, char_spans = data
    json_obj['answer_texts'] = answer_texts
    json_obj['passages_texts'] = passages_texts
    json_obj['qid'] = qid

    # question_tokens = tokenizer.tokenize(question_text)
    # passages_tokens = [tokenizer.tokenize(passage_text) for passage_text in passages_texts]

    passages_tokens = [passage_tokens[:max_passage_len] for passage_tokens in passages_tokens]
    question_tokens = question_tokens[:max_question_len]
    # if any([len(token.text) > max_num_characters for token in question_tokens]):
    #     return None
    # if any([len(token.text) > max_num_characters for sublist in passages_tokens for token in sublist]):
    #     return None
    char_spans = char_spans or []
    # We need to convert character indices in `passage_text` to token indices in
    # `passage_tokens`, as the latter is what we'll actually use for supervision.
    passages_offsets = [[(token.idx, token.idx + len(token.text)) for token in passage_tokens]
                        for passage_tokens in passages_tokens]
    token_spans = []
    if answer_texts:
        for passage_id, span_in_passage in enumerate(char_spans):
            passage_offsets = passages_offsets[passage_id]
            passage_token_spans = []
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


def process(l):
    # DuReader to MSMarco
    j = json.loads(l)
    j['query_id'] = j.pop('question_id')
    j['query'] = j.pop('question')
    passages = []
    # test data don't have these infomation
    try:
        j['query_type'] = j.pop('question_type')
        j.pop('fact_or_opinion')
        j.pop('entity_answers')
    except Exception as e:
        pass
    # char_only don't need jieba
    if char_only:
        j['question_tokens'] = segmented_text_to_tuples([ch for ch in j['query'].replace(' ', '')])
    else:
        j['question_tokens'] = segmented_text_to_tuples(j.pop('segmented_question'))
    for k in j['documents']:
        data = {}
        if 'is_selected' in k:
            if k['is_selected']:
                data['is_selected'] = 1
            else:
                data['is_selected'] = 0
        data['passage_text'] = ' '.join(k['paragraphs'])
        data['url'] = ''
        passages.append(data)
    if char_only:
        j['passages_tokens'] = [segmented_text_to_tuples([ch for ch in ''.join(doc['paragraphs'])])
                                for doc in j['documents']]
    else:
        j['passages_tokens'] = [segmented_text_to_tuples(sum(doc['segmented_paragraphs'], []))
                                for doc in j['documents']]
    j['passages'] = passages
    j.pop('documents')
    # ---------
    # find word span, if fuzzy_matching is true, this will find best f1 match
    # ---------
    instance = process_one_sample(j, fuzzy_matching)
    if instance is None:
        return None
    # word span to char span and convert to json_obj format
    json_obj = data_to_json_obj(instance)
    return json_obj


def parallel_process_file(file_name):
    if char_only:
        f_save = open(file_name + '.char.instances.' + str(p_idx), 'w')
    else:
        f_save = open(file_name + '.instances.' + str(p_idx), 'w')
    with open(file_name) as f:
        dureader_data = f.readlines()
    chunk_size = len(dureader_data) // 13 + 1
    dureader_data = dureader_data[chunk_size * p_idx: chunk_size * (p_idx + 1)]
    pool = Pool()
    for json_obj in pool.imap_unordered(process, dureader_data):
        if json_obj is not None:
            f_save.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    f_save.close()


def main():
    # data_path = "/home/2014010348"
    # file_name = os.path.join(data_path, 'zhidao.train.json.merge_passage')
    # parallel_process_file(file_name)
    data_path = "/home/2014010348/dureader/data"
    file_name = os.path.join(data_path, 'search.train.json.merge_passage')
    parallel_process_file(file_name)


if __name__ == '__main__':
    Token = namedtuple('Token', ['text', 'idx'])
    char_only = True
    drop_invalid = True
    max_passage_len = 500
    fuzzy_matching = False
    max_question_len = 50
    f1_or_rougeL = 'rougeL'
    max_num_characters = 30
    ip = socket.gethostbyname(socket.gethostname())
    p_idx = int(ip.split('.')[-1]) % 13
    # f1_or_rougeL = 'f1'
    main()
