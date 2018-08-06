import sys
sys.path.append('..')
import time
import numpy
from common.np import np


class Trainer:
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