import os
import numpy
from tqdm import tqdm
from collections import Counter


def load_data(file_name, max_vocab_size=50000, min_word_freq=3, seed=1984):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name
    src_data, tgt_data = [], []
    src_counter = Counter()
    tgt_counter = Counter()

    corpus_size = sum([1 for _ in open(file_path, 'r')])

    print('Loading corpus... (%s)' % file_path)
    with tqdm(total=corpus_size) as pbar:
        for line in open(file_path, 'r', encoding='utf8'):
            src, tgt = line.replace('\n', '').split('\t')
            src_words = src.split()
            tgt_words = ['<bos>'] + tgt.split() + ['<eos>']

            src_data.append(src_words)
            tgt_data.append(tgt_words)

            for word in src_words:
                src_counter[word] += 1

            for word in tgt_words:
                tgt_counter[word] += 1

            pbar.update(1)

    src_vocab = [w for w, f in src_counter.most_common(max_vocab_size) if f >= min_word_freq]
    tgt_vocab = [w for w, f in tgt_counter.most_common(max_vocab_size) if f >= min_word_freq]

    src_w2id = {w: i for i, w in enumerate(src_vocab + ['<ignore>'])}
    tgt_w2id = {w: i for i, w in enumerate(tgt_vocab + ['<ignore>'])}
    src_unk = len(src_w2id)
    tgt_unk = len(tgt_w2id)

    max_src = max([len(src) for src in src_data])
    max_tgt = max([len(tgt) for tgt in tgt_data])

    x = numpy.zeros((len(src_data), max_src), dtype=numpy.int)
    t = numpy.zeros((len(tgt_data), max_tgt), dtype=numpy.int)

    for i in tqdm(range(len(src_data))):
        src = numpy.array([src_w2id.get(w, src_unk) for w in src_data[i]])
        tgt = numpy.array([tgt_w2id.get(w, tgt_unk) for w in tgt_data[i]])
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

     # shuffle
    indices = numpy.arange(len(x))
    if seed is not None:
        numpy.random.seed(seed)
    numpy.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 1k test set
    test_set_num = 1000
    (x_test, x_train) = x[:test_set_num], x[test_set_num:]
    (t_test, t_train) = t[:test_set_num], t[test_set_num:]

    return (x_train, t_train), (x_test, t_test), (src_w2id, tgt_w2id), (src_id2w, tgt_id2w)

