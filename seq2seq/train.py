import os
import sys
sys.path.append('..')
import pickle
from common.dataset import load_data
from common.optimizer import Adam
from common.trainer import Trainer
from models import AttnBiSeq2Seq
from common.iterator import Iterator


# files
src_file = 'tanaka.ja.train'
tgt_file = 'tanaka.en.train'
save_dir = 'tanaka_ja_en'

# hyperparameter
max_vocab_size = 30000
min_word_freq = 2
wordvec_size = 300
hidden_size = 300
batch_size = 100
max_epoch = 30
eval_interval = 10
max_grad = 5.0

# dataset
(x_train, t_train), (x_test, t_test), (src_w2id, tgt_w2id, src_id2w, tgt_id2w) \
                            = load_data(src_file, tgt_file, max_vocab_size, min_word_freq)

# statistic
src_vocab_size = len(src_w2id)
tgt_vocab_size = len(tgt_w2id)

print('\n---', save_dir, '---')
print('src vocab size:', src_vocab_size)
print('tgt vocab size:', tgt_vocab_size)
print('train size:', len(x_train))
print('test size:', len(x_test))
print('\nwordvec size:', wordvec_size)
print('hidden size:', hidden_size)
print('batch size:', batch_size)
print('max epoch:', max_epoch)
print('max grad:', max_grad)
print()

# prepare
vocabs = { 'src_w2id': src_w2id, 'src_id2w': src_id2w,
           'tgt_w2id': tgt_w2id, 'tgt_id2w': tgt_id2w}
hyperparameters = {
    'max_vocab_size': max_vocab_size, 'min_word_freq': min_word_freq,
    'wordvec_size': wordvec_size, 'hidden_size': hidden_size,
    'batch_size': batch_size, 'max_epoch': max_epoch, 'max_grad': max_grad
    }
save_dir = os.getcwd() + '/' + save_dir
os.makedirs(save_dir, exist_ok=True)
with open(save_dir + '/vocabs.pkl', 'wb') as f:
    pickle.dump(vocabs, f)
with open(save_dir + '/hyperparameters.pkl', 'wb') as f:
    pickle.dump(hyperparameters, f)

# train
train_iter = Iterator(x_train, t_train, batch_size, max_epoch)
model = AttnBiSeq2Seq(src_vocab_size, tgt_vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer, save_dir)

trainer.report_bleu(x_test, t_test, vocabs)
trainer.run(train_iter, eval_interval, max_grad)
