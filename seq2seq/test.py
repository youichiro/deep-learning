import sys
sys.path.append('..')
import pickle
import numpy
from tqdm import tqdm
from models import AttnBiSeq2Seq, AttentionSeq2Seq, Seq2Seq
from common.bleu import compute_bleu

# import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = 'sans-serif'

# test data
src_test_file = '../datasets/naist/naist_gawonide.err.wkt'
tgt_test_file = '../datasets/naist/naist_gawonide.ans.wkt'

# load model
save_dir = 'mai_error100k_3'
model_file = 'e18-model.pkl'
vocabs_file = 'vocabs.pkl'
hyper_file = 'hyperparameters.pkl'

with open(save_dir + '/' + vocabs_file, 'rb') as f:
    vocabs = pickle.load(f)
with open(save_dir + '/' + hyper_file, 'rb') as f:
    hypers = pickle.load(f)
src_w2id = vocabs['src_w2id']
src_id2w = vocabs['src_id2w']
tgt_w2id = vocabs['tgt_w2id']
tgt_id2w = vocabs['tgt_id2w']
src_vocab_size = len(src_w2id)
tgt_vocab_size = len(tgt_w2id)
wordvec_size = hypers['wordvec_size']
hidden_size = hypers['hidden_size']

model = AttnBiSeq2Seq(src_vocab_size, tgt_vocab_size, wordvec_size, hidden_size)
model.load_params(save_dir + '/' + model_file)
bos_id = tgt_w2id['<bos>']
eos_id = tgt_w2id['<eos>']


def translate(model, src_words):
    ids = [src_w2id.get(w, src_w2id['<unk>']) for w in src_words]
    src = numpy.array([ids])
    predict = model.generate(src, bos_id=bos_id, eos_id=eos_id)
    out_words = [tgt_id2w[int(idx)] for idx in predict]
    return out_words


_idx = 0
def plot_attention(attention_map, row_labels, column_labels):
    fig, ax = plt.subplots()
    ax.pcolor(attention_map, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)
    ax.patch.set_facecolor('black')
    ax.set_yticks(numpy.arange(attention_map.shape[0])+0.5, minor=False)
    ax.set_xticks(numpy.arange(attention_map.shape[1])+0.5, minor=False)
    ax.invert_yaxis()
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)

    global _idx
    _idx += 1
    plt.show()


def visualize_attention(model, src_words, tgt_words):
    x = numpy.array([[src_w2id.get(w, src_w2id['<unk>']) for w in src_words]])
    t = numpy.array([[tgt_w2id.get(w, tgt_w2id['<unk>']) for w in tgt_words]])
    model.forward(x, t)
    d = model.decoder.attention.attention_weights
    d = numpy.array(d)
    attention_map = d.reshape(d.shape[0], d.shape[2])

    row_labels = [src_id2w[i] for i in x[0]]
    column_labels = [tgt_id2w[i] for i in t[0]]
    column_labels = column_labels[1:]
    plot_attention(attention_map, row_labels, column_labels)


def test(src_test_file, tgt_test_file):
    with open(src_test_file, 'r') as f:
        src_test_data = f.readlines()
    with open(tgt_test_file, 'r') as f:
        tgt_test_data = f.readlines()
    assert len(src_test_data) == len(tgt_test_data)

    print('model: {}/{}'.format(save_dir, model_file))
    print('test data: {}'.format(src_test_file))
    print('           {}\n'.format(tgt_test_file))

    reference_data = []
    translation_data = []
    for i in tqdm(range(len(src_test_data))):
        src_words = src_test_data[i].split()
        tgt_words = tgt_test_data[i].split()
        out_words = translate(model, src_words)
        out_words = [w for w in out_words if w != '<eos>']
        src = ' '.join(src_words)
        ref = ''.join(tgt_words)
        out = ''.join(out_words)
        bleu = compute_bleu([[tgt_words]], [out_words])[0]

        result = True if out == ref else False
        print('{}\tsrc\t{}\t{}'.format(i + 1, src, result))
        print('{}\tref\t{}\t{}'.format(i + 1, ref, result))
        print('{}\tout\t{}\t{}'.format(i + 1, out, result))
        print('\t\tbleu: {:.4f}'.format(bleu))

        reference_data.append([tgt_words])
        translation_data.append(out_words)

    total_bleu = compute_bleu(reference_data, translation_data, smooth=True)[0]
    print('BLEU: {:.4f}'.format(total_bleu))


if __name__ == '__main__':
    test(src_test_file, tgt_test_file)
    # example
    # src_sentence = '私 は テニス 部員 で す 。'
    # tgt_sentence = "i 'm in the tennis club ."
    # src_words = src_sentence.split()
    # tgt_words = tgt_sentence.split()

    # output = ' '.join(translate(model, src_words))
    # print('\ninput:', src_sentence)
    # print('output:', output)

    # visualize_attention(model, src_words, tgt_words)
