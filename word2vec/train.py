import sys
sys.path.append('..')
import numpy as np
from common.config import GPU
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.utils import create_contexts_target, get_vocab, to_device
from word2vec.iterator import WindowIterator


window_size = 3
hidden_size = 500
batch_size = 300
max_epoch = 20
max_vocab_size = 10000
min_word_freq = 1
eval_interval = 1000

corpus_file = '../datasets/testdata.txt'
# corpus_file = '/lab/ogawa/corpora/nikkei/nikkei_all.wakati_unidic.num2symbol_1M.txt'

print('\033[92m[ Hyper parameters ]\033[0m')
print('- window_size:', window_size)
print('- hidden_size:', hidden_size)
print('- batch_size:', batch_size)
print('- max_epoch:', max_epoch)
print('- max_vocab_size:', max_vocab_size)
print('- min_word_freq:', min_word_freq)
print('- corpus:', corpus_file)
print()

train, word_to_id, id_to_word = get_vocab(corpus_file, max_vocab_size, min_word_freq)
vocab_size = len(word_to_id)
unk_rate = train.count(word_to_id['UNK']) / len(train) * 100.0 if 'UNK' in word_to_id.keys() else 0.0

print('\n\033[92m[ statics ]\033[0m')
print('- token_size:', len(train))
print('- vocab_size:', vocab_size)
print('- unk_rate: {:.2f}%'.format(unk_rate))

train_iter = WindowIterator(train, window_size, batch_size, max_epoch)
model = CBOW(vocab_size, hidden_size, window_size, train)
optimizer = Adam()
trainer = Trainer(model, optimizer)

print('\n\033[92m[ progress ]\033[0m')
trainer.fit(train_iter, eval_interval=eval_interval)

word_vecs = model.word_vecs
if GPU:
    word_vecs = to_device(device=-1, x=word_vecs)

params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
print('\nsaved params to ' + pkl_file)
