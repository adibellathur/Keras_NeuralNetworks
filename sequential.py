import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class SequentialNetwork(object):

    def __init__(self):
        self.model = self.__load_sequential_model()
        self.features = None
        self.labels = None
        return

    def __load_sequential_model(self):
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', \
                      loss='binary_crossentropy', \
                      metrics=['accuracy'])
        return model

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

    def fit_model(self, epochs, batch_size, verbose=1):
        self.model.fit(self.features, \
                self.labels, \
                epochs=epochs, \
                batch_size=batch_size, \
                verbose=verbose)
        print('Accuracy: {}'.format(self.evaluate_accuracy()*100))
        return

    def evaluate_accuracy(self):
        _, accuracy = self.model.evaluate(self.features, self.labels)
        return accuracy

    def predict_on_features(self, features):
        return self.model.predict_classes(features)


def load_data(type):
    if type == 'diabetes':
        dataset = np.loadtxt('./data/pima-indians-diabetes.data.csv',
                            delimiter=',')
        X = dataset[:,0:8]
        y = dataset[:,8]
        return X, y
    return None, None

def main():
    sm = SequentialNetwork()
    X, y = load_data('diabetes')
    sm.load_dataset(X, y)
    sm.fit_model(epochs=150, batch_size=10)
    pred = sm.predict_on_features(X)
    # print("\n".join(['expected:{}, predicted: {}'.format(int(y[i]), pred[i][0]) \
    #                   for i in range(len(y))]))
    return

if __name__ == '__main__':
    main()
