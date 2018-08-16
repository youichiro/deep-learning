import sys
sys.path.append('..')
from datasets import sequence, dataset
from common.utils import to_gpu, eval_seq2seq
from common.bleu import compute_bleu
from common.optimizer import Adam
from common.trainer import Trainer
from models import AttentionSeq2Seq

dataset_file = 'tanaka_ja_en.train'
max_vocab_size = 10000

(x_train, t_train), (x_test, t_test), \
(src_w2id, tgt_w2id), (src_id2w, tgt_id2w) = dataset.load_data(dataset_file, max_vocab_size=max_vocab_size)

x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
x_train, t_train = to_gpu(x_train), to_gpu(t_train)
x_test, t_test = to_gpu(x_test), to_gpu(t_test)

print('src vocab_size:', len(src_w2id))
print('tgt vocab_size:', len(tgt_w2id))
print('train size:', len(x_train))

vocab_size = len(src_w2id) + len(tgt_w2id)
wordvec_size = 500
hidden_size = 300
batch_size = 300
max_epoch = 30
# max_grad = 5.0

model = AttentionSeq2Seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

for i in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, eval_interval=20)

    references = []
    translations = []

    for i in range(len(x_test)):
        src, tgt = x_test[[i]], t_test[[i]]
        tgt = tgt.flatten()
        verbose = i < 10
        start_id = tgt[0]
        tgt = tgt[1:]
        trainslation = model.generate(src, start_id, len(tgt))

        references.append([[tgt_id2w[int(c)] for c in tgt]])
        translations.append([tgt_id2w[int(c)] for c in trainslation])        

        # src = ''.join([src_id2w[int(c)] for c in src.flatten()[::-1]])
        # tgt = ' '.join([tgt_id2w[int(c)] for c in tgt])
        # guess = ' '.join([tgt_id2w[int(c)] for c in guess])

        # if verbose:
        #     print('src:', src)
        #     print('tgt:', tgt)
        #     print('out:', guess)
        #     print('---')

    blue_score = compute_bleu(references, translations, smooth=True)[0]
    print('BLEU: {:.4f}'.format(blue_score))

model.save_params()
