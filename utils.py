import numpy as np
import random

def eval_numerical_gradients(f, params, df, h=1e-5):
  grads = {p: np.zeros_like(p_vals) for p, p_vals in params.items()}
  for p, p_vals in params.items():
    it = np.nditer(p_vals, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
      ip = it.multi_index

      oldval = p_vals[ip]
      p_vals[ip] = oldval + h
      pos = f(**params).copy()
      p_vals[ip] = oldval - h
      neg = f(**params).copy()
      p_vals[ip] = oldval

      grads[p][ip] = np.sum((pos - neg) * df) / (2 * h)
      it.iternext()
  return grads

def relative_error(x, y, eps=1e-8):
  return np.max(np.abs(x - y) / (np.maximum(eps, np.abs(x) + np.abs(y))))

def grad_check_full(f, params, df, analytic_grads, h=1e-5):
  numerical_grads = eval_numerical_gradients(f, params, df, h)
  return {p: relative_error(analytic_grads[p], numerical_grads[p]) for p in params}

def grad_check_sparse(f, params, df, analytic_grads, num_trials=30, h=1e-5):
  errors = {}
  for p, p_vals in params.items():
    rel_errors_for_p = []
    for _ in range(num_trials):
      ip = tuple([random.randrange(m) for m in p_vals.shape])

      oldval = p_vals[ip]
      p_vals[ip] = oldval + h
      pos = f(**params)
      p_vals[ip] = oldval - h
      neg = f(**params)
      p_vals[ip] = oldval

      grad_numerical = np.sum((pos - neg) * df) / (2 * h)
      grad_analytic = analytic_grads[p][ip]
      rel_errors_for_p.append(relative_error(grad_analytic, grad_numerical))
    errors[p] = np.mean(rel_errors_for_p)
  return errors
