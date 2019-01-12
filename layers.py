import numpy as np
from utils import grad_check_full, grad_check_sparse
from functools import partial

def affine_forward(X, W, b):
  out = X.dot(W) + b
  cache = (X, W, b)
  return out, cache

def affine_backward(dout, cache):
  X, W, b = cache
  dW = X.T.dot(dout)
  dX = dout.dot(W.T)
  db = dout.sum(axis=0).T
  return dX, dW, db

def relu_forward(x):
  out = np.maximum(x, 0)
  cache = x
  return out, cache

def relu_backward(dout, cache):
  x = cache
  dx = np.zeros_like(x)
  dx[x > 0] = 1
  dx *= dout
  return dx

def sigmoid_forward(x):
  out = 1 / (1 + np.exp(-x))
  cache = out
  return out, cache

def sigmoid_backward(dout, cache):
  out = cache
  return out * (1-out) * dout

def softmax_loss(scores, y):
  nonpos_logits = scores - np.max(scores, axis=1, keepdims=True)
  exp_logit_sum = np.sum(np.exp(nonpos_logits), axis=1, keepdims=True)
  log_probs = nonpos_logits - np.log(exp_logit_sum)
  N = scores.shape[0]
  loss = -np.sum(log_probs[np.arange(N), y]) / N
  
  dx = np.exp(log_probs)
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

if __name__ == '__main__':
  # Test affine_backward:
  X = np.random.randn(10, 15)
  W = np.random.randn(15, 4)
  b = np.random.randn(4)
  dout = np.random.randn(10, 4)
  f = lambda **kwargs: affine_forward(**kwargs)[0]
  _, cache = affine_forward(X, W, b)
  analytic_grad = dict(zip(['X', 'W', 'b'], affine_backward(dout, cache)))
  for k, v in grad_check_full(f, {'W': W, 'X': X, 'b': b}, dout, analytic_grad).items():
    try:
      assert v < 1e-7
    except AssertionError:
      print('{} has relative error {}'.format(k, v))

  # Test relu_backward:
  x = np.random.randn(10, 100) * 4 - 2
  dout = np.random.randn(10, 100)
  f = lambda **kwargs: relu_forward(**kwargs)[0]
  _, cache = relu_forward(x)

  analytic_grad = relu_backward(dout, cache)
  analytic_grad = dict(zip(['x'], (analytic_grad,)))
  for k, v in grad_check_full(f, {'x': x}, dout, analytic_grad).items():
    try:
      assert v < 1e-7
    except AssertionError:
      print('{} has relative error {}'.format(k, v))

  # Test sigmoid_backward:
  x = np.random.randn(10, 100) * 4 - 2
  dout = np.random.randn(10, 100)
  f = lambda **kwargs: sigmoid_forward(**kwargs)[0]
  _, cache = sigmoid_forward(x)

  analytic_grad = sigmoid_backward(dout, cache)
  analytic_grad = dict(zip(['x'], (analytic_grad,)))
  for k, v in grad_check_full(f, {'x': x}, dout, analytic_grad).items():
    try:
      assert v < 1e-7
    except AssertionError:
      print('{} has relative error {}'.format(k, v))
