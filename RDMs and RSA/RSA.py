import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler

#1 Loadiing the Dataset
mnist = keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist

#2 Preprocessing
#2.1 Normalizing the data
x_train = x_train / 255
x_test = x_test / 255

#2.2 Reshaping the data
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#3 Building the model
#3.1 Defining the model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

#3.2 Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#3.3 Training the model
model.fit(x_train, y_train, epochs=5)

#3.4 Evaluating the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)