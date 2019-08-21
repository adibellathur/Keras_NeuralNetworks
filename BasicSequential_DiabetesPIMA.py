import numpy as np
from models.basic_sequential import BasicSequential

def load_data(type):
    if type == 'diabetes':
        dataset = np.loadtxt('./data/pima-indians-diabetes.data.csv',
                            delimiter=',')
        X = dataset[:,0:8]
        y = dataset[:,8]
        return X, y
    return None, None

def main():
    sm = BasicSequential()
    features, labels = load_data('diabetes')
    sm.load_dataset(features, labels)
    sm.fit_model(epochs=150, batch_size=10)
    pred = sm.predict_on_features(features)
    # print("\n".join(['expected:{}, predicted: {}'.format(int(y[i]), pred[i][0]) \
    #                   for i in range(len(y))]))
    return

if __name__ == '__main__':
    main()
