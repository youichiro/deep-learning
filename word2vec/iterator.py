import numpy as np


class WindowIterator:
    def __init__(self, dataset, window_size, batch_size, max_epoch, repeat=True):
        self.dataset =  np.array(dataset, np.int32)
        self.window_size = window_size
        self.batch_size = batch_size
        self.max_iters = len(dataset) // batch_size
        self.max_epoch = max_epoch
        self.repeat = repeat
        self.order = np.random.permutation(len(dataset) - window_size * 2).astype(np.int32)
        self.order += window_size
        self.current_position = 0
        self.iteration = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_new_epoch:
            self.epoch += 1

        if not self.repeat and self.epoch > 0:
            raise StopIteration
        elif self.repeat and self.epoch > self.max_epoch - 1:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i:i_end]
        offset = np.concatenate([np.arange(-self.window_size, 0),
                                 np.arange(1, self.window_size + 1)])
        pos = position[:, None] + offset[None,:]
        contexts = self.dataset.take(pos)
        target = self.dataset.take(position)

        if i_end >= len(self.order):
            np.random.shuffle(self.order)
            self.is_new_epoch = True
            self.current_position = 0
            self.iteration = 0
        else:
            self.iteration += 1
            self.is_new_epoch = False
            self.current_position = i_end

        return contexts, target
