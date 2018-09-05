import os
import sys
sys.path.append('..')
import pickle
import numpy
from common.dataset import load_data
from common.utils import calc_unk_ratio
from common.iterator import Iterator
from models import AttnBiSeq2Seq
from common.optimizer import Adam
from common.trainer import Trainer


# files
src_file = '../datasets/tanaka_corpus/train+dev.en'
tgt_file = '../datasets/tanaka_corpus/train+dev.ja'
save_dir = 'tanaka_en_ja'

# hyperparameter
max_vocab_size = 20000
min_word_freq = 1
max_len = 70
min_len = 3
dev_size = 500
wordvec_size = 300
hidden_size = 300
batch_size = 200
max_epoch = 20
eval_interval = 100
max_grad = None

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
status = {
    'src_file': src_file, 'tgt_file': tgt_file, 'train_size': len(x_train), 'dev_size': len(x_dev),
    'src_vocab_size': src_vocab_size, 'tgt_vocab_size': tgt_vocab_size,
    'src_unk': str(src_unk_ratio*100)[:5] + '%', 'tgt_unk': str(tgt_unk_ratio*100)[:5] + '%'
}
status.update(hyperparameters)
save_path = os.getcwd() + '/' + save_dir
os.makedirs(save_path, exist_ok=True)
with open(save_path + '/vocabs.pkl', 'wb') as f:
    pickle.dump(vocabs, f)
with open(save_path + '/hyperparameters.pkl', 'wb') as f:
    pickle.dump(hyperparameters, f)
with open(save_path + '/status.txt', 'w') as f:
    f.write('model: %s\n' % save_dir)
    for k, v in status.items():
        f.write('%s: %s\n' % (k, str(v)))

# print statistics and hyperparameters
print('\n---', save_dir, '---')
for k, v in status.items():
    print('%s: %s' %(k, str(v)))
print()

# train
train_iter = Iterator(x_train, t_train, batch_size, max_epoch)
model = AttnBiSeq2Seq(src_vocab_size, tgt_vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer, save_path)

trainer.report_bleu(x_dev, t_dev, vocabs)
trainer.run(train_iter, eval_interval, max_grad)
