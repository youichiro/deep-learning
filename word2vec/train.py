import sys
sys.path.append('..')
import numpy as np
from common.config import GPU
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.utils import create_contexts_target, get_vocab, to_cpu, to_gpu


window_size = 2
hidden_size = 100
batch_size = 30
max_epoch = 20
eval_interval = 5

corpus, word_to_id, id_to_word = get_vocab('../datasets/testdata.txt')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size, eval_interval=eval_interval)
trainer.plot()

word_vecs = model.word_vecs
if GPU:
    word_vecs = to_cpu(word_vecs)

params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)