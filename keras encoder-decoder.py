import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist


def create_model(x_train_noisy, X_train, x_test_noisy, X_test):
    global pixels

    model = Sequential()

    # ENCODER
    model.add(Dense(500, input_dim=pixels, activation='relu'))
    model.add(Dense(300, activation='relu'))

    # FLATTEN
    model.add(Dense(100, activation='relu'))

    # DECODER
    model.add(Dense(300, activation='relu'))
    model.add(Dense(500, activation='relu'))

    # Probability of each 10 classes
    model.add(Dense(784, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train_noisy, X_train, validation_data=(x_test_noisy, X_test), epochs=20, batch_size=250)
    return model, history


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # ADJUST DATA
    pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], pixels).astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255

    # Noise
    noise_factor = 0.333
    x_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    x_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    model, history = create_model(x_train_noisy, X_train, x_test_noisy, X_test)

    pred = model.predict(x_test_noisy)
    X_test = np.reshape(X_test, (10000, 28, 28)) * 255
    pred = np.reshape(pred, (10000, 28, 28)) * 255
    x_test_noisy = np.reshape(x_test_noisy, (-1, 28, 28)) * 255

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoches')
    plt.legend(['train', 'val'], loc='best')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoches')
    plt.legend(['train', 'val'], loc='best')
    plt.show()

    n = 5
    plt.figure(figsize=(10, 7))

    for i in range(n):
        # TOP TO BOTTOM ORDER:
        # plot original image
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot noisy image
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # plot predicted image
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(pred[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
