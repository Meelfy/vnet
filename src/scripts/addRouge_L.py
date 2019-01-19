import json
import pickle
import sys
from tqdm import tqdm
from multiprocessing import Pool
sys.path.append(".")
from src.utils import get_ans_by_f1


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
    qid, passages, query, answers = data
    question_text = query
    passage_texts = [passage['passage_text'] for passage in passages][:10]
    spans = []
    answer_texts = []
    flag_has_ans = False

    if len(passage_texts) != 10:
        passage_texts = passage_texts + [passage_texts[-1]] * (10 - len(passage_texts))
        # logger.info("the num of passage must be the same")
        # return None
    if 'No Answer Present.' in answers:
        # logger.info("No Answer Present.")
        # logger.info(answers)
        return None

    for passage_text in passage_texts:
        answers_in_passage = []
        span_in_passage = []

        flag_has_ans = get_em_ans(answers, passage_text, span_in_passage, answers_in_passage,
                                  flag_has_ans)
        if not flag_has_ans and len(answers) > 0:
            ans_rougeL = get_ans_by_f1(passage_text, answers)
            flag_has_ans = get_em_ans(ans_rougeL, passage_text, span_in_passage, answers_in_passage,
                                      flag_has_ans)
        answer_texts.append(answers)
        # answer_texts.append(answers_in_passage)
        spans.append(span_in_passage)
    if not flag_has_ans:
        # logger.info("ignore one 0 answer instance")
        # logger.info(answers)
        return None
    instance = (question_text, passage_texts, qid, answer_texts, spans)
    return instance


def add_rouge_read(file_path: str):
    with open(file_path) as f:
        source = json.load(f)
    instances = []
    dataset = ((qid, source['passages'][qid], source['query'][qid], source['answers'][qid])
               for qid in source['query_id'])
    pool = Pool()
    for instance in tqdm(pool.imap_unordered(process_one_sample, dataset)):
        # print(instance)
        if instance is not None:
            instances.append(instance)
    return instances


def main():
    file_path = '/data/nfsdata/meijie/data/msmarco/train_v2.1.json'
    instances = add_rouge_read(file_path)
    f_save = open(file_path + '.pickle', 'wb')
    pickle.dump(instances, f_save)
    f_save.close()

    f_reload = open(file_path + '.pickle', 'rb')
    instances_reload = pickle.load(f_reload)
    f_reload.close()

    for sample1, sample2 in tqdm(zip(instances, instances_reload)):
        assert sample1 == sample2


if __name__ == '__main__':
    main()
