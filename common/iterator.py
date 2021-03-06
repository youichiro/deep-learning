import numpy


class Iterator:
    def __init__(self, src_data, tgt_data, batch_size, max_epoch, repeat=True):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.max_iter = len(src_data) // batch_size + 1
        self.repeat = repeat
        self.order = numpy.random.permutation(len(src_data)).astype(numpy.int32)
        self.current_position = 0
        self.iteration = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_new_epoch:
            self.epoch += 1
            self.iteration = 0

        if not self.repeat and self.epoch > 0:
            raise StopIteration
        elif self.repeat and self.epoch > self.max_epoch - 1:
            raise StopIteration

        self.iteration += 1
        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i:i_end]

        batch_src = numpy.array([self.src_data[i] for i in position])
        batch_tgt = numpy.array([self.tgt_data[i] for i in position])

        if i_end >= len(self.order):
            numpy.random.shuffle(self.order)
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch_src, batch_tgt


class WindowIterator:
    def __init__(self, dataset, window_size, batch_size, max_epoch, repeat=True):
        self.dataset =  numpy.array(dataset, numpy.int32)
        self.window_size = window_size
        self.batch_size = batch_size
        self.max_iters = len(dataset) // batch_size
        self.max_epoch = max_epoch
        self.repeat = repeat
        self.order = numpy.random.permutation(len(dataset) - window_size * 2).astype(numpy.int32)
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
        offset = numpy.concatenate([numpy.arange(-self.window_size, 0),
                                    numpy.arange(1, self.window_size + 1)])
        pos = position[:, None] + offset[None,:]
        contexts = self.dataset.take(pos)
        target = self.dataset.take(position)

        if i_end >= len(self.order):
            numpy.random.shuffle(self.order)
            self.is_new_epoch = True
            self.current_position = 0
            self.iteration = 0
        else:
            self.iteration += 1
            self.is_new_epoch = False
            self.current_position = i_end

        return contexts, target
