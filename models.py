import numpy as np
import torch
import random
from collections import Counter
from tqdm import tqdm
import sklearn
from abc import ABC, abstractmethod
from layers import *

class Model(ABC):
  @abstractmethod
  def __str__(self):
    pass

  @abstractmethod
  def train(self):
    pass

  @abstractmethod
  def eval(self):
    pass

  def validate(self, X, y):
    return np.mean(self.eval(X) == y)

  def format_test(self, y_test, filename):
    with open(filename, 'w') as f:
      f.write('ImageId,Label\n')
      for i, y in enumerate(y_test):
        f.write('{},{}\n'.format(i+1, int(y)))

class KNNScratchModel(Model):
  def __init__(self, k):
    self.k = k
  
  def __str__(self):
    return 'KNN implemented from scratch with Numpy. Ties broken by choosing lowest class. Set k={}.'.format(self.k)

  def train(self, X, y):
    self.X_train = X
    self.y_train = y
  
  def eval(self, X):
    num_test = X.shape[0]
    test_sq = np.reshape(np.sum(X**2, axis=1), (-1, 1))
    test_train_prod = X.dot(self.X_train.T)
    train_sq = np.reshape(np.sum(self.X_train**2, axis=1), (1, -1))
    diffs = test_sq - 2 * test_train_prod + train_sq

    min_idx = np.argsort(diffs, axis=1)[:, :self.k]
    y_votes = self.y_train[min_idx]

    y_test = np.zeros(num_test, dtype=np.uint8)
    for i, votes in enumerate(y_votes):
      max_cnt = float('-inf')
      max_classes = []
      for cls, cnt in Counter(list(votes)).items():
        if cnt > max_cnt:
          max_classes = [cls]
          max_cnt = cnt
        elif cnt == max_cnt:
          max_classes.append(cls)
      y_test[i] = random.choice(max_classes)

    return y_test

class KNNSklearnModel(Model):
  def __init__(self, k):
    self.model = KNeighborsClassifier(n_neighbors=k)
    self.k = k
  
  def __str__(self):
    return 'KNN implemented with scikit-learn. Set k={}.'.format(self.k)

  def train(self, X, y):
    self.model.fit(X, y)
  
  def eval(self, X):
    self.model.predict(X)

  def validate(self, X, y):
    return self.model.score(X, y)

class FullyConnectedNNModel(Model):
  def __init__(self, input_dims, num_classes, hidden_sizes, learning_rate, reg_strength, batch_size, num_epochs):
    self.dims = [input_dims] + hidden_sizes + [num_classes]
    self.weights = [np.random.rand(i_dim, o_dim) * 0.001 for i_dim, o_dim in zip(self.dims[:-1], self.dims[1:])]
    self.biases = [np.zeros(o_dim) + 0.1 for o_dim in self.dims[1:]]
    self.learning_rate = learning_rate
    self.reg_strength = reg_strength
    self.batch_size = batch_size
    self.num_epochs = num_epochs

  def __str__(self):
    return 'FC-NN implemented from scratch with Numpy. Set alpha={}, lambda={}, batch size={}. Trained for {} epochs. Used architecture {} with sigmoid activations.'.format(*[
        self.learning_rate,
        self.reg_strength,
        self.batch_size,
        self.num_epochs,
        '-'.join(str(d) for d in self.dims),
      ])

  def loss(self, X, y=None):
    out = X
    affine_caches = []
    activation_caches = []
    for i in range(len(self.weights) - 1):
       out, cache = affine_forward(out, self.weights[i], self.biases[i])
       affine_caches.append(cache)
       out, cache = sigmoid_forward(out)
       activation_caches.append(cache)
    scores, final_cache = affine_forward(out, self.weights[-1], self.biases[-1])

    if y is None:
      return scores

    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg_strength * sum(np.sum(W**2) for W in self.weights)
    loss = data_loss + reg_loss

    weight_grads = [0] * len(self.weights)
    bias_grads = [0] * len(self.biases)

    dout, weight_grads[-1], bias_grads[-1] = affine_backward(dscores, final_cache)
    weight_grads[-1] += self.reg_strength * self.weights[-1]
    for i in reversed(range(len(self.weights) - 1)):
      dout = sigmoid_backward(dout, activation_caches[i])
      dout, weight_grads[i], bias_grads[i] = affine_backward(dout, affine_caches[i])

    return loss, weight_grads, bias_grads

  def train(self, X, y):
    y = y.astype(np.uint8)
    loss_history = []
    for _ in range(self.num_epochs):
      shuffled_idx = np.random.permutation(X.shape[0])
      X_shuffled = X.copy()[shuffled_idx]
      y_shuffled = y.copy()[shuffled_idx]
      X_split = np.array_split(X_shuffled, X_shuffled.shape[0] // self.batch_size + 1)
      y_split = np.array_split(y_shuffled, y_shuffled.shape[0] // self.batch_size + 1)
      for batch_X, batch_y in zip(X_split, y_split):
        loss, weight_grads, bias_grads = self.loss(batch_X, batch_y)
        loss_history.append(loss)
        for i in range(len(self.weights)):
          self.weights[i] -= self.learning_rate * weight_grads[i]
        for i in range(len(self.biases)):
          self.biases[i] -= self.learning_rate * bias_grads[i]
    return loss_history

  def eval(self, X):
    return np.argmax(self.loss(X), axis=1)
