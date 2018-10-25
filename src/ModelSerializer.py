import pickle


class ModelSerializer:
    def __init__(self, filename):
        self.file = filename
        self.model = None

    def serialize(self):
        with open(self.file, "wb") as f:
            pickle.dump(self.model, f)

    def deserialize(self):
        with open(self.file, "rb") as f:
            self.model = pickle.load(f)
        return self.model

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_file(self):
        return self.file

    def set_file(self, file):
        self.file = file
