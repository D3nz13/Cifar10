import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, BatchNormalization

def plot_classes(X_train, y_train, num_classes, classes_names):
    """
    This function plots the first occurance of each class
    """
    nrows, ncols = 4, 3
    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        index = np.where(y_train == i)[0][0]
        name = classes_names[i]
        plt.subplot(nrows, ncols, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_train[index])
        plt.title(name)
    plt.show()


def prepare_data(X_train, y_train, X_test, y_test, num_classes):
    """
    This function prepares the data so that it can be fed into a CNN
    """
    # encoding the target variable
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    # data type conversion
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalization
    X_train /= 255
    X_test /= 255

    return X_train, y_train, X_test, y_test


def create_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def make_plots(history):
    # plot accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.show()

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_classes = 10
    classes_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    # plot_classes(X_train, y_train, num_classes, classes_names)
    X_train, y_train, X_test, y_test = prepare_data(X_train, y_train, X_test, y_test, num_classes)
    cnn = create_model()
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = cnn.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    make_plots(history)
    # cnn.summary()