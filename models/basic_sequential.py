from keras.models import Sequential
from keras.layers import Dense

from .sequential import SequentialNetwork

class BasicSequential(SequentialNetwork):

    def __init__(self):
        super().__init__()
        self.model = self._load_sequential_model()

    def _load_sequential_model(self):
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', \
                      loss='binary_crossentropy', \
                      metrics=['accuracy'])
        return model

    def fit_model(self, epochs, batch_size, verbose=1):
        self.model.fit(self.features, \
                self.labels, \
                epochs=epochs, \
                batch_size=batch_size, \
                verbose=verbose)
        print('Accuracy: {}'.format(self.evaluate_accuracy()*100))
        return
