from keras.models import Sequential
from keras.layers import Dense
from abc import ABC, abstractmethod

class SequentialNetwork(ABC):

    def __init__(self):
        self.model = None
        self.features = None
        self.labels = None
        return

    @abstractmethod
    def _load_sequential_model(self):
        pass

    def set_features(self, features):
        self.features = features
        return

    def set_labels(self, labels):
        self.labels = labels
        return

    def load_dataset(self, features, labels):
        self.features = features
        self.labels = labels
        return

    @abstractmethod
    def fit_model(self, epochs, batch_size, verbose):
        pass

    def evaluate_accuracy(self):
        _, accuracy = self.model.evaluate(self.features, self.labels)
        return accuracy

    def predict_on_features(self, features):
        return self.model.predict_classes(features)
