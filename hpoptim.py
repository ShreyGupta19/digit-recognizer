import numpy as np
from tqdm import tqdm
import pathos.pools as pp
from functools import partial

NUM_CPUS = 4

class HPOptimizer:
  def __init__(self, model_type, X_train, y_train, percent_val):
    self.model_type = model_type
    X_data = X_train
    y_data = y_train

    num_train = int(X_data.shape[0] * (1-percent_val))

    indices = np.random.permutation(X_data.shape[0])
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    self.X_train, self.X_val = X_data[train_idx], X_data[val_idx]
    self.y_train, self.y_val = y_data[train_idx], y_data[val_idx]

  def random_sweep(self, num_trials, sweep_funcs, fixed_params=None):
    history = {}
    max_score = float('-inf')
    max_model = None
    for _ in tqdm(range(num_trials)):
      params = fixed_params if fixed_params is not None else {}
      random_params = {k: fn() for k, fn in sweep_funcs.items()}
      params.update(random_params)
      m = self.model_type(**params)
      m.train(self.X_train, self.y_train)
      combo_key = tuple(sorted(random_params.items()))
      history[combo_key] = m.validate(self.X_val, self.y_val)
      if history[combo_key] > max_score:
        max_score = history[combo_key]
        max_model = m
    return history, max_model

  def random_sweep_mp(self, num_trials, sweep_funcs, fixed_params=None):
    def worker(_):
      params = fixed_params if fixed_params is not None else {}
      random_params = {k: fn() for k, fn in sweep_funcs.items()}
      params.update(random_params)
      m = self.model_type(**params)
      m.train(self.X_train, self.y_train)
      combo_key = tuple(sorted(random_params.items()))
      acc = m.validate(self.X_val, self.y_val)
      return combo_key, acc, m

    pool = pp.ProcessPool(NUM_CPUS)
    results = []
    with tqdm(total=num_trials) as pbar:
      for res in pool.imap(worker, range(num_trials)):
        print('hey')
        results.append(res)
        pbar.update()
    pool.close()
    pool.join()

    max_score = float('-inf')
    max_model = None
    history = {}
    for combo_key, acc, model in results:
      history[combo_key] = acc
      if acc > max_score:
        max_score = acc
        max_model = model

    return history, max_model
