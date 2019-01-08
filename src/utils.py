import torch
from typing import List

import string
import re
import collections


def memory_effient_masked_softmax(vector: torch.Tensor, mask: torch.Tensor,
                                  dim: int = -1, mask_value=-1e7) -> torch.Tensor:
    """
    This is an approximate version of `allennlp.nn.util.masked_softmax`.
    By using less operations here than the original `masked_softmax`, we save a lot of memory.
    But you should be careful that this function does not return an array of ``0.0``, as the
    original `mask_softmax` does, in the case that the input vector is completely masked.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector + (1 - mask) * mask_value, dim=dim)
    return result


class MaxF1Mesure():
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(self, s):
        if not s:
            return []
        return self.normalize_answer(s).split()

    def compute_exact(self, a_gold, a_pred):
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

    def compute_f1(self, a_gold, a_pred):
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def calc_score(self, candidates, answers):
        candidates_text = candidates[0]
        max_f1 = 0
        for ans in answers:
            f1 = self.compute_f1(candidates_text, ans)
            max_f1 = max(max_f1, f1)
            if max_f1 == 1:
                break
        return f1


def get_answers_with_RougeL(passage, answers, threshold=0.7):
    token_as = [ans.split(' ') for ans in answers]
    token_p = passage.split(' ')
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
    # return [item['candidate'] for item in candidates]


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


def get_ans_by_f1(passage, answers, threshold=0.7):
    candidates = []
    measure = MaxF1Mesure()
    indices = [0] + [m.start() for m in re.finditer(' ', passage)]
    for i, lo in enumerate(indices):
        lo = lo + 1
        for hi in indices[i:] + list(set([indices[-1], len(passage)])):
            candidate = passage[lo:hi]
            if len(candidate) == 0:
                continue
            score = measure.calc_score([candidate], answers)
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
    return [item['candidate'] for item in candidates]
