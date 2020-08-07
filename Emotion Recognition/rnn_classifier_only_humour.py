import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'

    filenames = list()

    filenames += ['humour_'+group+'.txt']

    X = load_group(filenames, filepath)

    y = load_file(prefix + group + '/y_'+group+'.txt')

    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    X, y = load_dataset_group('train', prefix + 'EEGSetHumour/')
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.33)

    flat_list = [item for sublist in testy for item in sublist]

    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)

    # print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy,flat_list

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 100, 25
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # print(n_timesteps,n_features,n_outputs)
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.05))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    y_pred = model.predict_classes(testX, verbose=0)


    return accuracy,y_pred


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(repeats=10):
    scores = list()
    trainX, trainy, testX, testy, flat_list_y_true = load_dataset()
    for r in range(repeats):
        score,y_pred = evaluate_model(trainX, trainy, testX, testy)
        # print(flat_list_y_true)
        # print(y_pred)
        print(confusion_matrix(flat_list_y_true, y_pred))
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)




if __name__=="__main__":
    run_experiment()