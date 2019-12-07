import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from models.convolutional import ConvolutionalNeuralNetwork
import cv2

IMG_SIZE = (28,28,1)
CLASSES_DICT = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'}
GUESS_INDEX = 2

def load_data():
    (train_x, train_y), (test_x, test_y) =  mnist.load_data()
    train_x = train_x.astype('float32')
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    train_x /= 255
    train_y = to_categorical(train_y, 10)
    return train_x, train_y

def main():
    features_train, labels_train = load_data()
    cnn = ConvolutionalNeuralNetwork(IMG_SIZE, CLASSES_DICT, load_model_path='./snapshots/cnn_10.h5')
    cnn.set_features(features_train)
    cnn.set_labels(labels_train)
    # cnn.fit_model(10, 16)
    # cnn.save_model('./snapshots/cnn_10.h5')
    print(cnn.predict_on_image(features_train[GUESS_INDEX]))
    cv2.imshow('num?', features_train[GUESS_INDEX])
    cv2.waitKey()
    return

if __name__ == "__main__":
    main()
