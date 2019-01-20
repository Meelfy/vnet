import os
import json
import sys
from embedding_models.model import GlyphEmbedding
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from src.utils import get_ans_by_f1, get_answers_with_RougeL
# export PYTHONPATH=/home/meefly/working/WordLanguageModel/embedding_models/:$PYTHONPATH;
# export PYTHONPATH=/home/meefly/working/WordLanguageModel/glyph_embedding/:$PYTHONPATH

from glyph_embedding.utils.default_config import GlyphEmbeddingConfig
glyph_config = GlyphEmbeddingConfig()
glyph_embedding = GlyphEmbedding(glyph_config)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    file_path = '/data/nfsdata/meijie/data/msmarco/'
    data_samples = []
    with open(os.path.join(file_path, 'train_v2.1.json.instances'), 'r') as f:
        for idx in range(100):
            line = f.readline()
            data_samples.append(json.loads(line.strip()))
    # dict_keys(['answer_texts', 'passages_texts', 'qid', 'question_tokens',
    #            'passages_tokens', 'token_spans'])
    # data_samples[0].keys()
    json_obj = data_samples[1]
    for json_obj in data_samples:
        # passages_tokens = json_obj['passages_tokens']
        passages_texts = json_obj['passages_texts']
        answer_texts = json_obj['answer_texts']
        # token_spans = json_obj['token_spans']
        for idx in range(len(passages_texts)):
            # if [-1, -1] in token_spans[idx]:
            #     continue
            if answer_texts[idx][0] in passages_texts[idx]:
                continue
            golden_answers = list(set([ans[0] for ans in answer_texts]))
            rougeL_answer = get_answers_with_RougeL(passages_texts[idx], golden_answers)
            F1_answer = get_ans_by_f1(passages_texts[idx], golden_answers)
            if not rougeL_answer or not F1_answer:
                continue
            print('{s:{c}^{n}}'.format(s='start', n=50, c='-'))
            # print('{s:{c}^{n}}'.format(s='passages_tokens', n=50, c='-'))
            # print(passages_tokens[idx])

            print('{s:{c}^{n}}'.format(s='passages_texts', n=50, c='-'))
            print(passages_texts[idx])

            def highlight_text(text, color):
                f1_start_idx = passages_texts[idx].find(text)
                f1_end_idx = f1_start_idx + len(text)
                print(passages_texts[idx][:f1_start_idx] +
                      color + passages_texts[idx][f1_start_idx:f1_end_idx] +
                      bcolors.ENDC +
                      passages_texts[idx][f1_end_idx:])
            highlight_text(rougeL_answer[0], bcolors.OKGREEN)
            highlight_text(F1_answer[0], bcolors.OKBLUE)
            print('{s:{c}^{n}}'.format(s='golden_answer', n=50, c='-'))
            print(golden_answers)

            # print('{s:{c}^{n}}'.format(s='rougeL_answer', n=50, c='-'))
            # start = passages_tokens[idx][token_spans[idx][0][0]][1]
            # end = passages_tokens[idx][token_spans[idx][0][1]][1]
            # print(passages_texts[idx][start: end])

            print('{s:{c}^{n}}'.format(s='rougeL_answer', n=50, c='-'))
            print(rougeL_answer)

            print('{s:{c}^{n}}'.format(s='F1_answer', n=50, c='-'))
            print(F1_answer)

            # print('{s:{c}^{n}}'.format(s='answer_texts', n=50, c='-'))
            # print(answer_texts[idx][0])

            # print('{s:{c}^{n}}'.format(s='token_spans', n=50, c='-'))
            # print(token_spans[idx])

            print('{s:{c}^{n}}'.format(s='end', n=50, c='-'))
            print('\n')


if __name__ == '__main__':
    main()
