# encoding: utf-8
"""
@author: meijie
@contact: admin@mejiex.vip

@version: 1.0
@file: merge_paragraphs.py
@time: 2019年1月22日 16:11:43

按照 http://aclweb.org/anthology/D18-1235 对 DuReader 数据集进行预处理
"""
import json
import os
import numpy as np
from tqdm import tqdm_notebook as tqdm
from src.utils import ChineseMaxF1Mesure


def first_sentence(text, grain_size='sentence'):
    if text is None:
        return ''
    line_break = set('\n\r')
    delimiter = line_break | set('。？！；.?!;')
    sub_delimiter = delimiter | set('，,')

    if grain_size == 'phrase':
        selected_delimiter = sub_delimiter
    else:
        # grain_size = 'sentence'
        selected_delimiter = delimiter
    for idx in range(len(text)):
        if text[idx] in selected_delimiter:
            # 遇到 '.' 前后是数字或字母不切
            if text[idx] == '.' and (idx - 1 >= 0 and text[idx - 1].isalnum()) \
                    and (idx + 1 < len(text) and text[idx + 1].isalnum()):
                continue
            sentence = text[:idx + 1]
            if sentence == '':
                continue
            if sentence is None:
                return ''
            return sentence
    return ''


def merge_paragraphs(file_name, top_K=3):
    f_read = open(file_name)
    f_write = open(file_name + '.merge_passage', 'w')
    measure = ChineseMaxF1Mesure()
    for idx_l, line in tqdm(enumerate(f_read)):
        # if idx_l > 10:
        #     break
        raw_json = json.loads(line)
        for idx_doc, doc in enumerate(raw_json['documents']):
            passage_text = ''
            paragraph_score = []
            for paragraphs in doc['paragraphs']:
                refs = raw_json.get('answers') or [raw_json.get('question')]
                score = measure.calc_score([paragraphs], refs)
                paragraph_score.append(score)
            np.array(paragraph_score).argsort()[-top_K:][::-1]
            try:
                rank = sorted(list(np.array(paragraph_score).argsort()[-top_K:][::-1]))[0]
            except Exception as e:
                rank = 0
            passage_text += doc.get('title', '')[:20]
            passage_text += ''.join(doc['paragraphs'][rank: rank + 2])
            passage_text += ''.join([first_sentence(text) for text in doc['paragraphs'][rank + 2:]])
            doc['paragraphs'] = [passage_text]
            raw_json['documents'][idx_doc] = doc
        f_write.write(json.dumps(raw_json, ensure_ascii=False) + '\n')
    f_read.close()
    f_write.close()


if __name__ == '__main__':
    file_path = '/data/nfsdata/meijie/data/dureader/raw/trainset/'
    file_name = os.path.join(file_path, 'search.train.json')
    merge_paragraphs(file_name)
    file_path = '/data/nfsdata/meijie/data/dureader/raw/trainset/'
    file_name = os.path.join(file_path, 'zhidao.train.json')
    merge_paragraphs(file_name)
    file_path = '/data/nfsdata/meijie/data/dureader/raw/devset/'
    file_name = os.path.join(file_path, 'search.dev.json')
    merge_paragraphs(file_name)
    file_path = '/data/nfsdata/meijie/data/dureader/raw/devset/'
    file_name = os.path.join(file_path, 'zhidao.dev.json')
    merge_paragraphs(file_name)
    file_path = '/data/nfsdata/meijie/data/dureader/raw/testset/'
    file_name = os.path.join(file_path, 'search.test.json')
    merge_paragraphs(file_name)
    file_path = '/data/nfsdata/meijie/data/dureader/raw/testset/'
    file_name = os.path.join(file_path, 'zhidao.test.json')
    merge_paragraphs(file_name)
