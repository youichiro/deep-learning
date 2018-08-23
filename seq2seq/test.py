import pickle
import numpy
from models import AttnBiSeq2Seq

vocabs_file = 'vocabs.pkl'
with open(vocabs_file, 'rb') as f:
    vocabs = pickle.load(f)

src_w2id = vocabs['src_w2id']
tgt_w2id = vocabs['tgt_w2id']
tgt_id2w = vocabs['tgt_id2w']

sent = '私 は 彼 を 呼ん だ 。'
sent_ids = [src_w2id[word] for word in sent.split()]

model_file = 'AttnBiSeq2Seq.pkl'
model = AttnBiSeq2Seq(100, 100, 100, 100)
model.load_params(model_file)
predict = model.generate(numpy.array([sent_ids]), tgt_w2id['<eos>'])

output = ''.join([tgt_id2w[int(idx)] for idx in predict])
print(output)
