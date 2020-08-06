# Copyright (c) 2019 Tradeshift
# Copyright (c) 2020 Sarthak Mittal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
from invoicenet.common.model import Model
import numpy as np


def train(model: Model):
    n_updates = 50000
    print_interval = 20
    best = float("inf")

    start = time.time()
    for i in range(n_updates):
        train_loss = model.train_batch()
        if not np.isfinite(train_loss):
            raise ValueError("NaN loss")

        if i % print_interval == 0:
            took = time.time() - start
            val_loss = model.val_batch()
            print("%05d/%05d %f batches/s train loss: %f val loss: %f" % (i, n_updates, (i + 1) / took, train_loss, val_loss))
            if not np.isfinite(val_loss):
                raise ValueError("NaN loss")
            if val_loss < best:
                model.save("best")

    return model
