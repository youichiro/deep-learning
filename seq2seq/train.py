import sys
sys.path.append('..')
from datasets import sequence, dataset
from common.utils import to_gpu
from common.bleu import compute_bleu
from common.evaluator import eval_blue
from common.optimizer import Adam
from common.trainer import Trainer
from models import AttentionSeq2Seq, AttnBiSeq2Seq


dataset_file = 'tanaka_ja_en.train'
max_vocab_size = 10000

(x_train, t_train), (x_test, t_test), (src_w2id, tgt_w2id), (src_id2w, tgt_id2w) \
                                        = dataset.load_data(dataset_file, max_vocab_size)

x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
x_train, t_train = to_gpu(x_train), to_gpu(t_train)
x_test, t_test = to_gpu(x_test), to_gpu(t_test)

src_vocab_size = len(src_w2id)
tgt_vocab_size = len(tgt_w2id)

print('src vocab size:', src_vocab_size)
print('tgt vocab size:', tgt_vocab_size)
print('train size:', len(x_train))
print('test size:', len(x_test))

wordvec_size = 500
hidden_size = 300
batch_size = 300
max_epoch = 30
# max_grad = 5.0

# model = AttentionSeq2Seq(src_vocab_size, tgt_vocab_size, wordvec_size, hidden_size)
model = AttnBiSeq2Seq(src_vocab_size, tgt_vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

for i in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, eval_interval=20)

    eos_id = tgt_w2id['<eos>']
    for i in range(10):
        src, tgt = x_test[[i]], t_test[[i]]
        tgt = tgt.flatten()
        trainslation = model.generate(src, eos_id)

        src = ''.join([src_id2w[int(c)] for c in src.flatten()[::-1]])
        tgt = ' '.join([tgt_id2w[int(c)] for c in tgt])
        translation = ' '.join([tgt_id2w[int(c)] for c in trainslation])

        print('src:', src)
        print('tgt:', tgt)
        print('out:', translation)
        print('---')


    blue_score = eval_blue(model, x_test, t_test, tgt_id2w, tgt_w2id)
    print('BLEU: {:.4f}'.format(blue_score))

model.save_params()
