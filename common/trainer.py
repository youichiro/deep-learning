import os
import sys
sys.path.append('..')
import time
from common.np import np
from common.utils import to_gpu
from common.evaluator import eval_blue


class Trainer:
    def __init__(self, model, optimizer, save_dir):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0
        self.do_report_bleu = False

    def report_bleu(self, src_test, tgt_test, vocabs):
        self.do_report_bleu = True
        self.src_test = src_test
        self.tgt_test = tgt_test
        self.src_w2id, self.src_id2w = vocabs['src_w2id'], vocabs['src_id2w']
        self.tgt_w2id, self.tgt_id2w = vocabs['tgt_w2id'], vocabs['tgt_id2w']

    def culc_bleu(self, model):
        bleu_score = eval_blue(model, self.src_test, self.tgt_test, self.tgt_id2w, self.tgt_w2id)
        return bleu_score

    def save_model(self, model, epoch):
        model.save_params(self.save_dir + '/e' + str(epoch) + '-model.pkl')

    def run(self, iterator, eval_interval=20, max_grad=None):
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_tile = time.time()
        for batch_src, batch_tgt in iterator:
            batch_src = to_gpu(batch_src)
            batch_tgt = to_gpu(batch_tgt)

            loss = model.forward(batch_src, batch_tgt)
            model.backward()
            params, grads = remove_duplicate(model.params, model.grads)
            if max_grad is not None:
                clip_grads(grads, max_grad)
            optimizer.update(params, grads)
            total_loss += loss
            loss_count += 1

            if eval_interval and (iterator.iteration % eval_interval) == 0:
                avg_loss = total_loss / loss_count
                elapsed_time = time.time() - start_tile
                print('| epoch %d \t| iter %d / %d \t| time %d[s] \t| loss %.2f'
                        % (iterator.epoch + 1, iterator.iteration, iterator.max_iter, elapsed_time, avg_loss))
                self.loss_list.append(float(avg_loss))
                total_loss, loss_count = 0, 0

            if iterator.is_new_epoch and self.do_report_bleu:
                bleu_score = self.culc_bleu(model)
                print('bleu: %.4f' % bleu_score)

            if iterator.is_new_epoch:
                self.save_model(model, iterator.epoch + 1)
                print('Saved model.')


class Word2vecTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None

    def fit(self, iterator, eval_interval=20):
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for batch_contexts, batch_target in iterator:
            loss = model.forward(batch_contexts, batch_target)
            model.backward()
            params, grads = remove_duplicate(model.params, model.grads)
            optimizer.update(params, grads)
            total_loss += loss
            loss_count += 1

            if eval_interval and (iterator.iteration % eval_interval) == 0:
                avg_loss = total_loss / loss_count
                elapsed_time = time.time() - start_time
                print('| epoch %d \t| iter %d / %d \t| time %d[s] \t| loss %.2f \t|'
                            % (iterator.epoch + 1, iterator.iteration + 1, iterator.max_iters, elapsed_time, avg_loss))
                self.loss_list.append(float(avg_loss))
                total_loss, loss_count = 0, 0


def remove_duplicate(params, grads):
    """パラメータ配列中の重複する重みを一つに集約し、その重みに対応する勾配を加算する"""
    params, grads = params[:], grads[:]

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break
        if not find_flg: break
    return params, grads


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
