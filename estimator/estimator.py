'''
  TensorFlow provides a higher level estimator API with pre-built model to train and predict data
  - DNNClassifier
  - DNNLinearCombinedClassifier
  - DNNLinearCombinedRegressor
  - DNNRegressor
  - LinearClassifier
  - LinearRegressor
'''

import tensorflow as tf
import numpy as np

x_feature = tf.feature_column.numeric_column('f1')

# training
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x = {'f1': np.array([1., 2., 3., 4.])},
  y = np.array([1.5, 3.5, 5.5, 7.5]),
  batch_size = 2,
  num_epochs = None,
  shuffle = True
)

# testing
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x = {'f1': np.array([5., 6., 7.,])},
  y = np.array([9.5, 11.5, 13.5]),
  num_epochs = 1,
  shuffle = False
)

# prediction
samples = np.array([8., 9.])
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
  x = {'f1': samples},
  num_epochs = 1,
  shuffle = False
)

# define regressor
regressor = tf.estimator.LinearRegressor(
  feature_columns = [x_feature],
  model_dir = './output'
)

regressor.train(input_fn=train_input_fn, steps = 2500)

avg_loss = regressor.evaluate(input_fn=test_input_fn)['average_loss']
print(f'average loss in testing: {avg_loss:.4f}')

# prediction
predictions = list(regressor.predict(input_fn=predict_input_fn))

for input, p in zip(samples, predictions):
  v = p['predictions'][0]
  print(f'{input}->{v:.4f}')
