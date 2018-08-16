import sys
sys.path.append('..')
from datasets import sequence
from common.utils import to_gpu
from common.optimizer import Adam
from common.trainer import Trainer
from models import AttentionSeq2Seq


(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
x_train = to_gpu(x_train)
t_train = to_gpu(t_train)

vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

model = AttentionSeq2Seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(x_train, t_train, max_epoch=max_epoch, batch_size=batch_size, max_grad=max_grad)

model.save_params()
