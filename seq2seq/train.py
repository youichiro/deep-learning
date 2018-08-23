import os
import sys
sys.path.append('..')
import pickle
from common.dataset import load_data
from common.utils import to_gpu
from common.bleu import compute_bleu
from common.evaluator import eval_blue
from common.optimizer import Adam
from common.trainer import Trainer
from models import AttnBiSeq2Seq
from iterator import Iterator


# files
src_file = 'wmt16.1M.de'
tgt_file = 'wmt16.1M.en'
save_dir = 'wmt16_de_en_1M'

# hyperparameter
max_vocab_size = 30000
min_word_freq = 2
wordvec_size = 300
hidden_size = 300
batch_size = 300
max_epoch = 30
eval_interval = 100
max_grad = 5.0

# dataset
(x_train, t_train), (x_test, t_test), (src_w2id, tgt_w2id, src_id2w, tgt_id2w) \
                            = load_data(src_file, tgt_file, max_vocab_size, min_word_freq)

# statistic
src_vocab_size = len(src_w2id)
tgt_vocab_size = len(tgt_w2id)
vocabs = { 'src_w2id': src_w2id, 'src_id2w': src_id2w,
           'tgt_w2id': tgt_w2id, 'tgt_id2w': tgt_id2w }

print('\nsrc vocab size:', src_vocab_size)
print('tgt vocab size:', tgt_vocab_size)
print('train size:', len(x_train))
print('test size:', len(x_test))
print()

# train
train_iter = Iterator(x_train, t_train, batch_size, max_epoch)
model = AttnBiSeq2Seq(src_vocab_size, tgt_vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.report_translation(x_test, t_test, vocabs)
trainer.run(train_iter, eval_interval, max_grad)

# save
os.makedirs(save_dir, exist_ok=True)
model.save_params(save_dir + '/model.pkl')
with open(save_dir + '/vocabs.pkl', 'wb') as f:
    pickle.dump(vocabs, f)

