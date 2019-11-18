class Model:
    def train_batch(self):
        raise NotImplementedError()

    def val_batch(self):
        raise NotImplementedError

    def load(self, name):
        raise NotImplementedError

    def save(self, name):
        raise NotImplementedError
