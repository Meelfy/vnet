# manage MSMARCO from passage to whitespace token
# author: ypt@pku.edu.cn
# 2019.2.1
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from typing import List
import numpy as np


class LabelGenerator(object):
    def __init__(self,
                 language: str = "en_core_web_sm"):
        self._wordSplliter = SpacyWordSplitter(language=language)

    def split_passage(self,
                      passage: str):  # 'str' -> [str]
        return self._wordSplliter.split_words(passage.lower())

    def gen_gold_span(self,
                      passage: str,
                      answers: List[str],
                      candidate_num: int = 1,
                      extend_length: int = 20,  # 处理候选span两边扩展的长度(答案长度大于20时)
                      roughl_threshold: float = 0.7  # 超过此阈值的才会被认为是best span
                      ):
        # 返回best_span=
        # [start, end], best_span_roughl, passage_split, best_span_text 如果没有符合要求的 [], 0, passage_split, ''
        """
        1 tokenize passage/answers
        2 get candidate spans by finding LCSubstr
        3 extend the candidate spans
        4 find best span from candidate spans by computing rough-L

        :param passage: 'I'm a passage'
        :param answers: ['I'm answer one', 'I'm answer two']
        :param candidate_num: number of candidate spans
        :param extend_length: if len(answer) < 20 : candidate spans are extended to len(candidate span)
                              + 2 * extend_length else: extended to 3 * len(candidate span)
        :param roughl_threshold: best span must have a rough-L value that > roughl_threshould
        :return:  best_span=[start, end], best_span_roughl, passage_split, best_span_text 如果没有符合要求的 [],
                  0, passage_split, ''
        """
        # split passages and answers
        passage_split_spacy = self.split_passage(passage)
        passage_split = []
        for passage_spacy in passage_split_spacy:
            passage_split.append(passage_spacy.text)
        answers_split = []
        for answer in answers:
            answer_split_spacy = self.split_passage(answer)
            answer_split = []
            for answer_spacy in answer_split_spacy:
                answer_split.append(answer_spacy.text)
            answers_split.append(answer_split)

        # gen Candidate span
        start_idx = []
        end_idx = []
        recall_answers = []  # 记录每个标准答案在文章中最大字串的recall
        for answer_split in answers_split:
            start_idx_, end_idx_ = self.get_LCSubstr(passage_split, answer_split)
            if (len(start_idx_) == 0 and len(end_idx_) == 0):
                continue
            start_idx.append(start_idx_)
            end_idx.append(end_idx_)
            recall_answers.append((end_idx_[0] - start_idx_[0] + 1) / len(answer_split))
        if len(recall_answers) == 0:
            return [], 0, passage_split, ''
        answer_idx = np.argmax(recall_answers)
        answer_split = answers_split[answer_idx]  # 接下来只关注有最大召回的答案
        candidate_length = min(len(start_idx[answer_idx]), candidate_num)  # 确保candidate num不大于最长字串数目
        candidate_start_idx = start_idx[answer_idx][0:candidate_length]
        candidate_end_idx = end_idx[answer_idx][0:candidate_length]

        # extend Candidate span to gen gold span
        best_span = []
        best_span_roughl = 0
        for i in range(candidate_length):
            if (candidate_end_idx[i] - candidate_start_idx[i] + 1) < 20:
                extend_length = candidate_end_idx[i] - candidate_start_idx[i] + 1
            start_idx_ = max(0, candidate_start_idx[i] - extend_length)
            end_idx_ = min(len(passage_split) - 1, candidate_end_idx[i] + extend_length)
            best_span_, best_span_roughl_ = self.get_answers_with_RougeL(
                passage_split[start_idx_:(end_idx_ + 1)],
                answer_split, roughl_threshold, start_idx_)

            if best_span_roughl_ > best_span_roughl:
                best_span_roughl = best_span_roughl_
                best_span = best_span_

        # output
        if len(best_span) > 0:
            best_span_text = passage_split[best_span[0]:(best_span[1] + 1)]
        else:
            best_span_text = ''
        return best_span, best_span_roughl, passage_split, best_span_text

    def get_answers_with_RougeL(self, token_p, token_a, threshold, offset):
        max_score = 0
        best_span = []
        for lo in range(len(token_p)):
            for hi in range(lo, len(token_p)):
                lcs = self.get_lcs(token_a, token_p[lo:(hi + 1)])

                score = self.get_rouge_l(lcs, len(token_a), hi - lo + 1)
                if score > threshold and score > max_score:
                    max_score = score
                    best_span = [lo, hi]
        if not best_span:
            return best_span, 0
        best_span[0] += offset
        best_span[1] += offset
        return best_span, max_score

    @staticmethod
    def get_LCSubstr(passage, answer):  # 返回start_id[] end_id[] 按lcs长度排序 个数为有相同字符的子串
        len_passage = len(passage)
        len_answer = len(answer)
        record = [[0 for _ in range(len_answer + 1)] for _ in range(len_passage + 1)]
        for i in range(len_passage):
            for j in range(len_answer):
                if (passage[i] == answer[j]):
                    record[i + 1][j + 1] = record[i][j] + 1
        record = np.array(record).reshape(-1)
        argsort_ = record.argsort()
        argend = argsort_ // (len_answer + 1) - 1
        argstart = argend - record[argsort_] + 1
        output_len = (record > 0).sum()
        return list(argstart[::-1][0:output_len]), list(argend[::-1][0:output_len])

    @staticmethod
    def get_lcs(token_a, token_p):
        lcs_map = [[0 for _ in range(0, len(token_p) + 1)]
                   for _ in range(0, len(token_a) + 1)]
        for j in range(1, len(token_p) + 1):
            for i in range(1, len(token_a) + 1):
                if (token_a[i - 1] == token_p[j - 1]):
                    lcs_map[i][j] = lcs_map[i - 1][j - 1] + 1
                else:
                    lcs_map[i][j] = max(lcs_map[i - 1][j], lcs_map[i][j - 1])
        return lcs_map[-1][-1]

    @staticmethod
    def get_rouge_l(lcs, answer_length, span_length):
        beta = 1.2
        rec = lcs / answer_length
        prec = lcs / span_length
        if (prec != 0 and rec != 0):
            score = ((1 + beta ** 2) * prec * rec) / float(rec + beta ** 2 * prec)
        else:
            score = 0.0
        return score


if __name__ == '__main__':
    a = 'ads fdsf sds dsd d, we %, ds daf kwef dddd' * 10
    b = [', dd ds daf kwef', 'fdsf sds d']
    label_generator = LabelGenerator()
    span, score, p, p2 = label_generator.gen_gold_span(a, b, roughl_threshold=0.7)
    print(span)
    print(score)
    print(p)
    print(p2)
