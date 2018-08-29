import os
import sys
sys.path.append('..')
import pickle
from common.dataset import load_data
from common.utils import calc_unk_ratio
from common.iterator import Iterator
from models import AttnBiSeq2Seq
from common.optimizer import Adam
from common.trainer import Trainer


# files
src_file = 'mai_error/mai2000.100k.err.wkt'
tgt_file = 'mai_error/mai2000.100k.ans.wkt'
save_dir = 'mai_error100k'

# hyperparameter
max_vocab_size = 40000
min_word_freq = 1
wordvec_size = 300
hidden_size = 300
batch_size = 100
max_epoch = 20
eval_interval = 50
max_grad = 10.0

# dataset
(x_train, t_train), (x_dev, t_dev), (src_w2id, tgt_w2id, src_id2w, tgt_id2w) \
                            = load_data(src_file, tgt_file, max_vocab_size, min_word_freq)
src_unk_ratio = calc_unk_ratio(x_train, src_w2id.get('<unk>', None), src_w2id.get('<ignore>', None))
tgt_unk_ratio = calc_unk_ratio(t_train, tgt_w2id.get('<unk>', None), tgt_w2id.get('<ignore>', None))

# statistic
src_vocab_size = len(src_w2id)
tgt_vocab_size = len(tgt_w2id)

print('\n---', save_dir, '---')
print('train size:', len(x_train))
print('dev size:', len(x_dev))
print('src vocab size:', src_vocab_size)
print('tgt vocab size:', tgt_vocab_size)
print('src unknown ratio: {:.2f}%'.format(src_unk_ratio * 100))
print('tgt unknown ratio: {:.2f}%'.format(tgt_unk_ratio * 100))
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

trainer.report_bleu(x_dev, t_dev, vocabs)
trainer.run(train_iter, eval_interval, max_grad)
