import os
import re
import numpy
import mojimoji
from tqdm import tqdm
from collections import Counter


digit_pattern = re.compile(r'(\d( \d)*)+')


def clean_text(text):
    text = mojimoji.zen_to_han(text, kana=False)
    text = digit_pattern.sub('#', text)
    return text



def load_data(src_file, tgt_file, max_vocab_size=50000, min_word_freq=3, max_len=40, min_len=4, dev_size=1000):
    src_data, tgt_data = [], []
    src_counter, tgt_counter = Counter(), Counter()

    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = f.readlines()

    print('Loading corpus... (%s and %s)' % (src_file, tgt_file))
    for src, tgt in zip(tqdm(src_lines), tgt_lines):
        src_words = src.replace('\n', '').split()
        # src_words = clean_text(src).replace('\n', '').split()
        tgt_words = tgt.replace('\n', '').split()
        # tgt_words = clean_text(tgt).replace('\n', '').split()
        tgt_words = ['<bos>'] + tgt_words + ['<eos>']

        if not min_len <= len(src_words) <= max_len or not min_len <= len(tgt_words) <= max_len:
            continue

        src_data.append(src_words)
        tgt_data.append(tgt_words)
        for w in src_words:
            src_counter[w] += 1
        for w in tgt_words:
            tgt_counter[w] += 1

    src_vocab = [w for w, f in src_counter.most_common(max_vocab_size) if f >= min_word_freq]
    tgt_vocab = [w for w, f in tgt_counter.most_common(max_vocab_size) if f >= min_word_freq]

    src_w2id = {w: i for i, w in enumerate(src_vocab + ['<ignore>'])}
    tgt_w2id = {w: i for i, w in enumerate(tgt_vocab + ['<ignore>'])}
    src_unk = len(src_w2id)
    tgt_unk = len(tgt_w2id)

    max_src = max([len(src) for src in src_data])
    max_tgt = max([len(tgt) for tgt in tgt_data])

    x = numpy.zeros((len(src_data), max_src), dtype=numpy.int32)
    t = numpy.zeros((len(tgt_data), max_tgt), dtype=numpy.int32)

    for i in tqdm(range(len(src_data))):
        src = numpy.array([src_w2id.get(w, src_unk) for w in src_data[i]], dtype=numpy.int32)
        tgt = numpy.array([tgt_w2id.get(w, tgt_unk) for w in tgt_data[i]], dtype=numpy.int32)
        x[i] = numpy.pad(src, (0, max_src-len(src)), 'constant', constant_values=src_w2id['<ignore>'])
        t[i] = numpy.pad(tgt, (0, max_tgt-len(tgt)), 'constant', constant_values=tgt_w2id['<ignore>'])

    for src in x:
        if src_unk in src:
            src_w2id['<unk>'] = src_unk
            break

    for tgt in t:
        if tgt_unk in tgt:
            tgt_w2id['<unk>'] = tgt_unk
            break

    src_id2w = {v: k for k, v in src_w2id.items()}
    tgt_id2w = {v: k for k, v in tgt_w2id.items()}

    x_dev, x_train = x[:dev_size], x[dev_size:]
    t_dev, t_train = t[:dev_size], t[dev_size:]
    return (x_train, t_train), (x_dev, t_dev), (src_w2id, tgt_w2id, src_id2w, tgt_id2w)
