import sys
sys.path.append('..')
from datasets import sequence, dataset
from common.utils import to_gpu, eval_seq2seq
from common.optimizer import Adam
from common.trainer import Trainer
from models import AttentionSeq2Seq

dataset_file = 'tanaka_ja_en_000.1000.train'
max_vocab_size = 50000

(x_train, t_train), (x_test, t_test), \
(src_w2id, tgt_w2id), (src_id2w, tgt_id2w) = dataset.load_data(dataset_file, max_vocab_size=max_vocab_size)

# (x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
# char_to_id, id_to_char = sequence.get_vocab()

x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
x_train, t_train = to_gpu(x_train), to_gpu(t_train)
x_test, t_test = to_gpu(x_test), to_gpu(t_test)

vocab_size = len(src_w2id) + len(tgt_w2id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

model = AttentionSeq2Seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

for i in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    for i in range(len(x_test)):
        src, tgt = x_test[[i]], t_test[[i]]
        tgt = tgt.flatten()
        verbose = i < 10
        start_id = tgt[0]
        tgt = tgt[1:]
        guess = model.generate(src, start_id, len(tgt))

        src = ''.join([src_id2w[int(c)] for c in src.flatten()])
        tgt = ''.join([tgt_id2w[int(c)] for c in tgt])
        guess = ''.join([tgt_id2w[int(c)] for c in guess])

        if verbose:
            src = src[::-1]
            print('src:', src)
            print('tgt:', tgt)
            print('out', guess)
            print('---')

# acc_list = []
# for i in range(max_epoch):
#     trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)
#     correct_num = 0
#     for i in range(len(x_test)):
#         question, correct = x_test[[i]], t_test[[i]]
#         verbose = i < 10
#         correct_num += eval_seq2seq(model, question, correct,
#                                     id_to_char, verbose, is_reverse=True)

#     acc = float(correct_num) / len(x_test)
#     acc_list.append(acc)
#     print('val acc %.3f%%' % (acc * 100))

model.save_params()
