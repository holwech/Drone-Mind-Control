import numpy as np


class RingBuffer2D:
    def __init__(self, width, length):
        if width == 1:
            print("For 1D RingBuffer use 'RingBuffer' instead")
            raise ValueError
        self.index = 0
        self.data = np.zeros((width, length), dtype='f')
        self.data_length = length

    def extend(self, x):
        x_index = (self.index + np.arange(len(x[0, :]))) % self.data_length
        self.data[:, x_index] = x
        self.index = x_index[-1] + 1

    def append(self, x):
        self.data[:, self.index] = x
        self.index = (self.index + 1) % self.data_length

    def get(self, data_row='all'):
        index = (self.index + np.arange(self.data_length)) % self.data_length
        if data_row == 'all':
            return self.data[:, index]
        else:
            return self.data[data_row, index]


class RingBuffer:
    def __init__(self, length):
        self.index = 0
        self.data = np.zeros(length, dtype='f')
        self.data_length = length

    def extend(self, x):
        x_index = (self.index + np.arange(len(x[0, :]))) % self.data_length
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def append(self, x):
        self.data[self.index] = x
        self.index = (self.index + 1) % self.data_length

    def get(self):
        index = (self.index + np.arange(self.data_length)) % self.data_length
        return self.data[index]
