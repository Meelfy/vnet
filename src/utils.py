import torch

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
