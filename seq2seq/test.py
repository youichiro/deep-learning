import pickle
import numpy
from models import AttnBiSeq2Seq

save_dir = 'tanaka_ja_en'
model_file = 'e30-model.pkl'
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

sent = '誰 が 一番 に 着 く か 私 に は 分か り ま せ ん 。'
sent_ids = [src_w2id[word] for word in sent.split()]
src = numpy.array([sent_ids])

model = AttnBiSeq2Seq(src_vocab_size, tgt_vocab_size, wordvec_size, hidden_size)
model.load_params(save_dir + '/' + model_file)
predict = model.generate(src, eos_id=tgt_w2id['<eos>'])

output = ' '.join([tgt_id2w[int(idx)] for idx in predict])
print(sent)
print(output)

