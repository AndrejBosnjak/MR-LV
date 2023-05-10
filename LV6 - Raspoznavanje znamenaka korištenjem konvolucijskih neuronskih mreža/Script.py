from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,28,28,1)
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape(10000,28,28,1)
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)

model1 = models.Sequential()

model1.add(layers.Conv2D(64, (3,3), activation="relu", strides=(1,1), input_shape=(28, 28, 1)))
model1.add(layers.Conv2D(32, (3,3), activation="relu", strides=(1,1)))
model1.add(layers.Flatten())
model1.add(layers.Dense(10, activation="softmax"))
model1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model1.summary()

model2 = models.Sequential()
model2.add(layers.Conv2D(64, (5,5), strides=(1,1), input_shape=(28,28,1)))
model2.add(layers.MaxPool2D((2,2)))
model2.add(layers.Conv2D(32, (5,5), strides=(1,1), input_shape=(28,28,1)))
model2.add(layers.MaxPool2D((2,2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(10, activation="softmax"))
model2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model2.summary()

mlp = MLPClassifier(hidden_layer_sizes=(100), activation="logistic", alpha=0.00001, solver="adam", max_iter=100)

model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
