import os
import numpy
from collections import Counter


src_w2id = {}
src_id2w = {}
tgt_w2id = {}
tgt_id2w = {}


def load_data(file_name, max_vocab_size=50000, min_word_freq=0, seed=1984):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name
    src_sentences, tgt_sentences = [], []
    src_counter = Counter()
    tgt_counter = Counter()

    for line in open(file_path, 'r', encoding='utf8'):
        src, tgt = line.replace('\n', '').split('\t')
        src_words = src.split()
        tgt_words = ['<bos>'] + tgt.split() + ['<eos>']

        src_sentences.append(src_words)
        tgt_sentences.append(tgt_words)

        for word in src_words:
            src_counter[word] += 1

        for word in tgt_words:
            tgt_counter[word] += 1

    src_vocab = [w for w, f in src_counter.most_common(max_vocab_size) if f > min_word_freq]
    tgt_vocab = [w for w, f in tgt_counter.most_common(max_vocab_size) if f > min_word_freq]

    src_w2id = {w: i for i, w in enumerate(src_vocab + [' '])}
    tgt_w2id = {w: i for i, w in enumerate(tgt_vocab + [' '])}
    src_unk = len(src_w2id)
    tgt_unk = len(tgt_w2id)

    max_src = max([len(src) for src in src_sentences])
    max_tgt = max([len(tgt) for tgt in tgt_sentences])

    x = numpy.zeros((len(src_sentences), max_src), dtype=numpy.int)
    t = numpy.zeros((len(tgt_sentences), max_tgt), dtype=numpy.int)

    for i, src in enumerate(src_sentences):
        pad_src = [src[j] if j < len(src) else ' ' for j in range(max_src)]
        x[i] = [src_w2id.get(w, src_unk) for w in pad_src]

    for i, tgt in enumerate(tgt_sentences):
        pad_tgt = [tgt[j] if j < len(tgt) else ' ' for j in range(max_tgt)]
        t[i] = [tgt_w2id.get(w, tgt_unk) for w in pad_tgt]

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

    # 1% for test set
    split_at = len(x) - len(x) // 100
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)


def get_vocab():
    return (src_w2id, tgt_w2id), (src_id2w, tgt_id2w)


def get_symbols():
    src_symbols = {
        '<bos>': src_w2id.get('<bos>', None),
        '<eos>': src_w2id.get('<eos>', None),
        '<unk>': src_w2id.get('<unk>', None),
    }
    tgt_symbols = {
        '<bos>': tgt_w2id.get('<bos>', None),
        '<eos>': tgt_w2id.get('<eos>', None),
        '<unk>': tgt_w2id.get('<unk>', None),
    }
    return src_symbols, tgt_symbols
