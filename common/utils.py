import re
import mojimoji
from collections import defaultdict
import numpy
from common.np import np
from tqdm import tqdm
from chainer.dataset.convert import to_device

digit_pattern = re.compile(r'(\d( \d)*)+')

def clean_ja_text(text):
    text = text.replace('\n', '')
    text = mojimoji.zen_to_han(text, kana=False)
    text = digit_pattern.sub('#', text)
    return text

def clean_en_text(text):
    text = text.replace('\n', '')
    text = text.lower()
    text = text.replace('.', ' .')
    text = text.replace('?', ' ?')
    text = text.replace(';', ' ;')
    text = text.replace(':', ' :')
    return text

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = clean_ja_text(line)
            words = text.split(' ')
            words.append('<eos>')
            yield words

def get_vocab(file_path):
    word_to_id = defaultdict(lambda: len(word_to_id))
    corpus = []
    corpus_size = sum(1 for line in open(file_path))
    print('loading corpus')
    with tqdm(total=corpus_size) as pbar:
        for words in load_corpus(file_path):
            [word_to_id[word] for word in words]
            corpus += [word_to_id[word] for word in words]
            pbar.update(1)
    id_to_word = {v: k for k, v in word_to_id.items()}
    word_to_id = dict(word_to_id)
    return corpus, word_to_id, id_to_word

def create_contexts_target(corpus, window_size=5):
    target = corpus[window_size: - window_size]
    contexts = [[corpus[idx + window]
                 for window in range(-window_size, window_size + 1) if window != 0]
                for idx in range(window_size, len(corpus) - window_size)]
    return np.array(contexts), np.array(target)

def cos_similarity(x, y):
    nx = x / (numpy.sqrt(np.sum(x ** 2)) + 1e-8)
    ny = y / (numpy.sqrt(np.sum(y ** 2)) + 1e-8)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print("'{}' is not found.".format(query))
        return
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    vocab_size = len(word_to_id)
    similarities = numpy.array([cos_similarity(word_matrix[i], query_vec) for i in range(vocab_size)])
    count = 0
    for i in (-1 * similarities).argsort():
        if id_to_word[i] == query:
            continue
        print("{}: {}".format(id_to_word[i], similarities[i]))
        count += 1
        if count >= top:
            return

def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)

def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)

