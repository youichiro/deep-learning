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
src_file = 'mai_error/mai2000.70k.err.wkt'
tgt_file = 'mai_error/mai2000.70k.ans.wkt'
save_dir = 'mai_error70k'

# hyperparameter
max_vocab_size = 40000
min_word_freq = 1
max_len = 40
min_len = 4
dev_size = 1000
wordvec_size = 300
hidden_size = 300
batch_size = 200
max_epoch = 20
eval_interval = 50
max_grad = 10.0

# dataset
(x_train, t_train), (x_dev, t_dev), (src_w2id, tgt_w2id, src_id2w, tgt_id2w) \
            = load_data(src_file, tgt_file, max_vocab_size, min_word_freq, max_len, min_len, dev_size)
src_vocab_size = len(src_w2id)
tgt_vocab_size = len(tgt_w2id)
src_unk_ratio = calc_unk_ratio(x_train, src_w2id.get('<unk>', None), src_w2id.get('<ignore>', None))
tgt_unk_ratio = calc_unk_ratio(t_train, tgt_w2id.get('<unk>', None), tgt_w2id.get('<ignore>', None))

# prepare
vocabs = { 'src_w2id': src_w2id, 'src_id2w': src_id2w,
           'tgt_w2id': tgt_w2id, 'tgt_id2w': tgt_id2w}
hyperparameters = {
    'max_vocab_size': max_vocab_size, 'min_word_freq': min_word_freq,
    'max_len': max_len, 'min_len': min_len,
    'wordvec_size': wordvec_size, 'hidden_size': hidden_size,
    'batch_size': batch_size, 'max_epoch': max_epoch, 'max_grad': max_grad
    }
statistics = {
    'src_file': src_file, 'tgt_file': tgt_file, 'train_size': len(x_train), 'dev_size': len(x_dev),
    'src_vocab_size': src_vocab_size, 'tgt_vocab_size': tgt_vocab_size,
    'src_unk': str(src_unk_ratio*100)[:5] + '%', 'tgt_unk': str(tgt_unk_ratio*100)[:5] + '%'
}
statistics.update(hyperparameters)
save_path = os.getcwd() + '/' + save_dir
os.makedirs(save_path, exist_ok=True)
with open(save_path + '/vocabs.pkl', 'wb') as f:
    pickle.dump(vocabs, f)
with open(save_path + '/hyperparameters.pkl', 'wb') as f:
    pickle.dump(hyperparameters, f)

# print statistics and hyperparameters
print('\n---', save_dir, '---')
for k, v in statistics.items():
    print('%s: %s' %(k, str(v)))
print()

# train
train_iter = Iterator(x_train, t_train, batch_size, max_epoch)
model = AttnBiSeq2Seq(src_vocab_size, tgt_vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer, save_path)
trainer.open_score_file(save_dir, **statistics)

trainer.report_bleu(x_dev, t_dev, vocabs)
trainer.run(train_iter, eval_interval, max_grad)
