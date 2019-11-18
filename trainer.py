import time
from model import Model
import numpy as np


def train(model: Model):
    n_updates = 50000
    print_interval = 20
    best = float("inf")

    start = time.time()
    for i in range(n_updates):
        loss = model.train_batch()
        if not np.isfinite(loss):
            raise ValueError("NaN loss")

        if i % print_interval == 0:
            took = time.time() - start
            print("%05d/%05d %f batches/s %f loss" % (i, n_updates, (i + 1) / took, loss))
            loss = model.val_batch()
            if not np.isfinite(loss):
                raise ValueError("NaN loss")
            if loss < best:
                model.save("best")

    return model
