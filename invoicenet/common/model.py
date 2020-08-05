class Model:
    def train_step(self):
        raise NotImplementedError()

    def val_step(self):
        raise NotImplementedError

    def load(self, name):
        raise NotImplementedError

    def save(self, name):
        raise NotImplementedError
