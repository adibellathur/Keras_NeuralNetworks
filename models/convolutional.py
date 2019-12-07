import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import losses
from keras import optimizers

from .sequential import SequentialNetwork

class ConvolutionalNeuralNetwork(SequentialNetwork):

    def __init__(self, img_size, classes_dict, load_model_path=None):
        super().__init__()
        self.img_size = img_size
        self.classes_dict = classes_dict
        if load_model_path:
            self.model = self._load_model_from_path(load_model_path)
        else:
            self.model = self._load_sequential_model()
        return

    def _load_sequential_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, \
                kernel_size=(2,2),
                strides=(1,1), \
                activation='relu', \
                input_shape=self.img_size))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Conv2D(filters=64, \
                kernel_size=(5,5), \
                strides=(1,1), \
                activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(len(self.classes_dict), activation='softmax'))
        model.compile(loss='categorical_crossentropy', \
                optimizer='adam', \
                metrics=['accuracy'])
        return model

    def _load_model_from_path(self, path):
        return load_model(path)

    def fit_model(self, epochs, batch_size, verbose=1):
        self.model.fit(self.features, \
                self.labels, \
                epochs=epochs, \
                batch_size=batch_size, \
                verbose=verbose)
        print('Accuracy: {}'.format(self.evaluate_accuracy()*100))
        return

    def predict_on_image(self, img):
        pred = self.model.predict(img.reshape(1, img.shape[0], img.shape[1], 1))
        return pred.argmax()

    def save_model(self, path):
        self.model.save(path)
        return
