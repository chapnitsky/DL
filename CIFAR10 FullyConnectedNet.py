from matplotlib import pylab as plt
# from google.colab import drive
import numpy as np
import pickle
import os


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = path
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    # Package data into a dictionary
    return X_train, y_train, X_val, y_val, X_test, y_test


path = os.getcwd()
# drive.mount('/content/drive')
x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_(x):
    return sigmoid(x) * (1 - sigmoid(x))


def firstquestion():
    inputLayerSize, hiddenLayerSize, outputLayerSize = 3, 3, 1
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
    Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))
    epochs = 3000

    for i in range(epochs):
        L1 = X.dot(Wh)
        # print(L1.shape == (4, 3))

        H = sigmoid(L1)
        # print(H.shape == (4, 3))

        L2 = H.dot(Wz)
        # print(L2.shape == (4, 1))

        Z = sigmoid(L2)
        # print(Z.shape == (4, 1))

        E = Y - Z
        # print(E.shape == (4, 1))

        dZ = E * sigmoid_(L2)
        # print(dZ.shape == (4, 1))

        dH = dZ.dot(Wz.T) * sigmoid_(L1)
        # print(dH.shape == (4, 3))

        Wz += np.dot(H.T, dZ)
        # print(Wz.shape == (3, 1))

        Wh += np.dot(X.T, dH)
        # print(Wh.shape == (3, 3))

    print(Z[0] < 0.05)  # what have we learnt?
    print(Z[1] > 0.95)  # what have we learnt?
    print(Z[2] > 0.95)  # what have we learnt?
    print(Z[3] < 0.05)  # what have we learnt?
    print(sigmoid(-10) < 6e-4)
    print(sigmoid(10) - 0.9999 < 6e-4)
    print(sigmoid(0) == 0.5)
    print(sigmoid_(0) == 0.25)


class TwoLayerNet(object):
    # ---------- constructor -----------------------------------------------
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.w1 = std * np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = std * np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    # ------------ -------------------------------------------
    def relu(self, x):
        return np.where(x < 0, 0, x)

    def relu_(self, x):
        return np.where(x < 0, 0, 1)

    def predict(self, X):
        y2, h1, y1 = self.forward(X)
        y_pred = np.argmax(y2, axis=1)
        return y_pred

    def forward(self, X):
        y1 = np.dot(X, self.w1) + self.b1
        h1 = self.relu(y1)
        y2 = np.dot(h1, self.w2) + self.b2
        return y2, h1, y1

    # --------------------------------------------------------------------
    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        W1, b1 = self.w1, self.b1
        W2, b2 = self.w2, self.b2

        for it in range(num_iters):
            print('iteration %d / ' % it)
            X_batch = None
            y_batch = None
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]

            loss, grads = self.lossAndGrad(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            self.w1 += -learning_rate * grads['W1']
            self.w2 += -learning_rate * grads['W2']
            self.b1 += -learning_rate * grads['b1']
            self.b2 += -learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print(f'iteration {it} / {num_iters}: loss {loss}')
            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                learning_rate *= learning_rate_decay
        return loss_history, train_acc_history, val_acc_history

    def computeLoss(self, NetOut, y):
        top = np.exp(NetOut)
        bottom = np.sum(top, axis=1, keepdims=True)
        div = top / bottom
        softmax_loss = -np.log(div)
        loss_to_return = softmax_loss[range(softmax_loss.shape[0]), y].mean()
        return loss_to_return

    def backPropagation(self, NetOut, h1, y1, X, y, reg):
        grads = {}
        W1, b1 = self.w1, self.b1
        W2, b2 = self.w2, self.b2
        N, D = X.shape

        exp_scores = np.exp(NetOut)
        dy2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]
        dy2[range(N), y] -= 1
        dy2 /= N
        # y2 = h1.dot(W2) + b2
        dW2 = h1.T.dot(dy2)
        # chain rule
        dh1 = dy2.dot(W2.T)
        # gradient of Relu
        dy1 = dh1 * (y1 >= 0)
        # y1 = X.dot(W1) + b1
        dW1 = X.T.dot(dy1)
        db1 = np.sum(dy1, axis=0)
        db2 = np.sum(dy2, axis=0)
        # Regularization
        dW1 += reg * W1
        dW2 += reg * W2

        grads['W1'] = dW1
        grads['W2'] = dW2

        grads['b1'] = db1
        grads['b2'] = db2
        return grads

    def lossAndGrad(self, X, y=None, reg=0.0):
        W1, b1 = self.w1, self.b1
        W2, b2 = self.w2, self.b2
        N, D = X.shape
        NetOut, h1, y1 = self.forward(X)
        if y is None:
            return NetOut
        loss = self.computeLoss(NetOut, y)
        # print(f'loss before : {loss}\n')
        reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss += reg_loss
        # backPropagation
        # print(f'loss after: {loss}\n')
        grads = self.backPropagation(NetOut, h1, y1, X, y, reg)
        return loss, grads


# def init_toy_data():
#     np.random.seed(1)
#     X = 10 * np.random.randn(num_inputs, input_size)
#     y = np.array([0, 1, 2, 2, 1])
#     return X, y


# def init_toy_model():
#     np.random.seed(0)
#     return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


if __name__ == "__main__":
    # firstquestion()
    input_size = 32 * 32 * 3
    hidden_size = 32
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # num_inputs = 4
    # net = init_toy_model()
    # X, y = init_toy_data()
    # los = net.lossAndGrad(X)
    # print('Your scores:')
    # print(los)
    # print()
    # print('correct scores:')
    # correct_los = np.asarray([
    #     [-0.81233741, -1.27654624, -0.70335995],
    #     [-0.17129677, -1.18803311, -0.47310444],
    #     [-0.51590475, -1.01354314, -0.8504215],
    #     [-0.15419291, -0.48629638, -0.52901952],
    #     [-0.00618733, -0.12435261, -0.15226949]])
    # print(correct_los)
    # print()
    # print('Difference between losses should be very smaller then 0.000001:')
    # print(np.sum(np.abs(los - correct_los)))
    los, trai, val = net.train(x_train, y_train, x_val, y_val, num_iters=5000)
    test_acc = (net.predict(x_test) == y_test).mean()
    train_acc = (net.predict(x_train) == y_train).mean()
    val_acc = (net.predict(x_val) == y_val).mean()
    print(f'val_acc:{val_acc}\ntest_acc:{test_acc}\ntrain_acc:{train_acc}')
    plt.plot(los)
    plt.title("loss")
    plt.show()
    plt.plot(trai)
    plt.title("train")
    plt.show()
    plt.plot(val)
    plt.title('validation')
    plt.show()