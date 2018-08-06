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


window_size = 2
hidden_size = 100
batch_size = 30
max_epoch = 30
eval_interval = 1
corpus_file = '../datasets/testdata.txt'

print('[ Hyper parameters ]')
print('- window_size:', window_size)
print('- hidden_size:', hidden_size)
print('- batch_size:', batch_size)
print('- corpus:', corpus_file)
print()

train, word_to_id, id_to_word = get_vocab(corpus_file)
vocab_size = len(word_to_id)
print('\n[ statics ]')
print('- token_size:', len(train))
print('- vocab_size:', vocab_size)
print()

train_iter = WindowIterator(train, window_size, batch_size, max_epoch)
model = CBOW(vocab_size, hidden_size, window_size, train)
optimizer = Adam()
trainer = Trainer(model, optimizer)

print('[ progress ]')
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

