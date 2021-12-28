import numpy as np
from matplotlib import pylab as plt
from sklearn.linear_model import LogisticRegression



def sigmoid(wx):
    return 1.0/(np.exp(-wx) + 1)


def log_likelihood(x, y, w):
    return np.sum(y*np.dot(x,w) - np.log10(1 + np.exp(np.dot(x, w))))


def LR(x, y, steps, l_rate, add_interxept=True):
    if add_interxept:
        inter = np.ones((x.shape[0], 1))
        x = np.hstack((inter, x))
    w = np.zeros(x.shape[1])
    for step in range(steps):
        wx = np.dot(x, w)
        pred = np.round(sigmoid(wx)).astype(np.int32)
        err = y - pred
        grad = np.dot(x.T, err)
        w += l_rate*grad

        if step % 10000 == 0:
            print(f'likelihood:\n{log_likelihood(x, y, w)}')

    return w


np.random.seed(12)
num_obs = 5000
x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_obs)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_obs)
x = np.vstack((x1, x2)).astype(np.float32)
y = np.hstack((np.zeros(num_obs), np.ones(num_obs)))

plt.figure(figsize=(12, 4))
plt.scatter(x[:, 0], x[:, 1], c=y, alpha=.4)
plt.show()

weights = LR(x, y, steps=10000, l_rate=5e-5)  # mine

clf = LogisticRegression()  # ski
clf.fit(x, y)
print(f'inter:\n{clf.intercept_}\ncoef:\n{clf.coef_}')
print(f'weights are:\n{weights}')
data_with_intercept = np.hstack((np.ones((x.shape[0], 1)), x))
final_scores = np.dot(data_with_intercept, weights)
preds = np.round(sigmoid(final_scores))
print('Accuracy from scratch: {0}'.format((preds == y).sum().astype(float) / len(preds)))
print('Accuracy from sk-learn: {0}'.format(clf.score(x, y)))
plt.figure(figsize=(12, 8))
plt.scatter(x[:, 0], x[:, 1],
c=(preds == y) - 1, alpha=.8, s=50)
plt.show()