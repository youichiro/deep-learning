import sys
sys.path.append('..')
import numpy
from common.utils import cos_similarity
from common.bleu import compute_bleu


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


def eval_seq2seq(model, question, correct, id_to_char, verbos=False, is_reverse=False):
    correct = correct.flatten()
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])

    if verbos:
        if is_reverse:
            question = question[::-1]

        print('Q', question)
        print('T', correct)
        if correct == guess:
            print('[ok]', guess)
        else:
            print('[ng]', guess)
        print('---')

    return 1 if guess == correct else 0


def eval_blue(model, x_test, t_test, vocabs):
    src_w2id, src_id2w = vocabs['src_w2id'], vocabs['src_id2w']
    tgt_w2id, tgt_id2w = vocabs['tgt_w2id'], vocabs['tgt_id2w']
    bos_id = tgt_w2id['<bos>']
    eos_id = tgt_w2id['<eos>']
    empty_id = tgt_w2id['<ignore>']
    references = []
    translations = []

    for i in range(len(x_test)):
        src, tgt = x_test[[i]], t_test[[i]]
        tgt = tgt.flatten()
        translation = model.generate(src, eos_id)

        r = [[tgt_id2w[int(i)] for i in tgt if int(i) not in [bos_id, eos_id, empty_id]]]
        references.append(r)

        t = [tgt_id2w[int(i)] for i in translation if int(i) not in [bos_id, eos_id, empty_id]]
        translations.append(t)

    src = ' '.join([src_id2w[int(i)] for i in x_test[0] if int(i) != src_w2id['<ignore>']])
    print('src: ', src)
    print('ref: ' + ' '.join(references[0][0]))
    print('out: ' + ' '.join(translations[0]))
    score = compute_bleu(references, translations, smooth=True)
    return score[0]

