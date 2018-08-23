import numpy


class Iterator:
    def __init__(self, src_data, tgt_data, batch_size, max_epoch, repeat=True):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.max_iter = len(src_data) // batch_size
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
        
        if not self.repeat and self.epoch > 0:
            raise StopIteration
        elif self.repeat and self.epoch > self.max_epoch - 1:
            raise StopIteration
        
        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i:i_end]

        batch_src = [self.src_data[i] for i in position]
        batch_tgt = [self.tgt_data[i] for i in position]

        if i_end >= len(self.order):
            numpy.random.shuffle(self.order)
            self.is_new_epoch = True
            self.current_position = 0
            self.iteration = 0
        else:
            self.iteration += 1
            self.is_new_epoch = False
            self.current_position = i_end

        return batch_src, batch_tgt
