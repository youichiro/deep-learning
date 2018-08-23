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


src_file = 'wmt16.1M.de'
tgt_file = 'wmt16.1M.en'
max_vocab_size = 30000
min_word_freq = 3
wordvec_size = 300
hidden_size = 300
batch_size = 300
max_epoch = 30
eval_interval = 100
max_grad = 5.0

(x_train, t_train), (x_test, t_test), (src_w2id, tgt_w2id), (src_id2w, tgt_id2w) \
                            = load_data(src_file, tgt_file, max_vocab_size, min_word_freq)

src_vocab_size = len(src_w2id)
tgt_vocab_size = len(tgt_w2id)
vocabs = { 'src_w2id': src_w2id, 'src_id2w': src_id2w,
           'tgt_w2id': tgt_w2id, 'tgt_id2w': tgt_id2w }

print('src vocab size:', src_vocab_size)
print('tgt vocab size:', tgt_vocab_size)
print('train size:', len(x_train))
print('test size:', len(x_test))

train_iter = Iterator(x_train, t_train, batch_size, max_epoch)
model = AttnBiSeq2Seq(src_vocab_size, tgt_vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.report_translation(x_test, t_test, vocabs)
trainer.run(train_iter, eval_interval, max_grad)

# for i in range(max_epoch):
#     trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, eval_interval=20)

#     eos_id = tgt_w2id['<eos>']
#     for i in range(10):
#         src, tgt = x_test[[i]], t_test[[i]]
#         tgt = tgt.flatten()
#         trainslation = model.generate(src, eos_id)

#         src = ' '.join([src_id2w[int(c)] for c in src.flatten()]).replace('<ignore>', '')
#         tgt = ' '.join([tgt_id2w[int(c)] for c in tgt]).replace('<ignore>', '')
#         translation = ' '.join([tgt_id2w[int(c)] for c in trainslation])

#         print('src:', src)
#         print('tgt:', tgt)
#         print('out:', translation)
#         print('---')


#     blue_score = eval_blue(model, x_test, t_test, tgt_id2w, tgt_w2id)
#     print('BLEU: {:.4f}'.format(blue_score))

model.save_params('naist_model.pkl')

with open('vocabs.pkl', 'wb') as f:
    pickle.dump(vocabs, f)

