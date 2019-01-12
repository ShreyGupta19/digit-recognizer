from models import KNNScratchModel, KNNSklearnModel, FullyConnectedNNModel
from hpoptim import HPOptimizer
import time
import numpy as np
import random

def extract_data(train_file, test_file):
  print('Importing data...')
  train_data = np.loadtxt(train_file, delimiter=',', skiprows=1)
  X_train = train_data[:,1:]
  y_train = train_data[:,0]
  X_test = np.loadtxt(test_file, delimiter=',', skiprows=1)
  print('Import complete.')
  return X_train, y_train, X_test

def run_knn_hp_tuning(X_train, y_train, X_test, model_type, out_file):
  print('Tuning Hyperparameters...')
  hp = HPOptimizer(KNNScratchModel, X_train, y_train, 0.2)
  history, best_model = hp.random_sweep_mp(20, {'k': lambda: random.randint(1, 25)})
  for combo, acc in sorted(history.items(), key=lambda x: x[1], reverse=True): 
    print('k = {}: {}'.format(combo[0][1], acc))
  print('Evaluating best model...')
  best_model.format_test(best_model.eval(X_test), out_file)
  print(best_model)

def run_fcnn_hp_tuning(X_train, y_train, X_test, out_file):
  print('Tuning Hyperparameters...')
  hp = HPOptimizer(FullyConnectedNNModel, X_train, y_train, 0.2)
  history, best_model = hp.random_sweep(80, {
      'learning_rate': lambda: 10**np.random.uniform(-4, -1),
      'reg_strength': lambda: 10**np.random.uniform(-5, -2)
    }, fixed_params={
      'input_dims': X_train.shape[1],
      'num_classes': 10,
      'hidden_sizes': [300, 100],
      'batch_size': 64,
      'num_epochs': 3,
    })
  print(len(history))
  for combo, acc in sorted(history.items(), key=lambda x: x[1], reverse=True)[:10]: 
    print('lr = {}, reg = {}: {}'.format(combo[0][1], combo[1][1], acc))
  print('Evaluating best model...')
  best_model.format_test(best_model.eval(X_test), out_file)
  print(best_model)

def run_fcnn(X_train, y_train, X_test, out_file):
  model = FullyConnectedNNModel(X_train.shape[1], 10, [300], 1e-4, 2.5e-2, 64, 3)
  loss_history = model.train(X_train, y_train)
  for i in range(0, len(loss_history), 20):
    print(loss_history[i])
  model.format_test(model.eval(X_test), out_file)
  print(model)

if __name__ == '__main__':
  X_train, y_train, X_test = extract_data('data/train.csv', 'data/test.csv')
  run_knn_hp_tuning(X_train, y_train, X_test, KNNSklearnModel, './knn-sklearn-hpt-out.csv')
  # run_fcnn_hp_tuning(X_train, y_train, X_test, './fcnn-scratch-out.csv')
