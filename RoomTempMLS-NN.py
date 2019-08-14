import time
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import math
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy
from tensorflow.keras.callbacks import TensorBoard

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

dense_layers = [0, 1, 2, 3]
layer_sizes = [8, 16, 32, 64, 128]
look_backs = [10, 24, 48, 48*2]


dataframe = pandas.read_csv('C:\\Projects\\Tensorflow\\output.csv',
                            usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
print(dataset.shape)

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for look_back in look_backs:
            NAME = "{}-lookback-{}-nodes-{}-dense-{}".format(look_back,
                                                             layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir="C:/Projects/Tensorflow/logs/{}".format(NAME))

            # fix random seed for reproducibility
            numpy.random.seed(7)
            # load the dataset

            # print(dataset)
            # split into train and test sets
            train_size = int(len(dataset) * (6/7))
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
            #print(len(train), len(test))
            # convert an array of values into a dataset matrix

            def create_dataset(dataset, look_back):
                dataX, dataY = [], []
                for i in range(len(dataset)-look_back-1):
                    a = dataset[i:(i+look_back), 0]
                    dataX.append(a)
                    dataY.append(dataset[i + look_back, 0])
                return numpy.array(dataX), numpy.array(dataY)

            # reshape into X=t and Y=t+1
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(test, look_back)
            # create and fit Multilayer Perceptron model
            model = Sequential()
            for l in range(dense_layer-1):
                model.add(Dense(layer_size, input_dim=look_back, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
            model.fit(trainX, trainY, epochs=20, batch_size=2, verbose=2, callbacks=[tensorboard])
            # evaluate the out of sample data with model
            val_loss, val_acc = model.evaluate(testX, testY)
            print(val_loss)  # model's loss (error)
            print(val_acc)  # model's accuracy
