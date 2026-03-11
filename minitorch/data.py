import numpy as np


class DataLoader:
    def __init__(self, x, y, batch_size=32, shuffle=True, drop_last=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n = x.shape[0]

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            if self.drop_last and (end - start) < self.batch_size:
                break
            idx = indices[start:end]
            yield self.x[idx], self.y[idx]

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size
