import tensorflow as tf
import pandas as pd
import argparse

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
  train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
  test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
  return train_path, test_path

def load_data(y_name='Species'):
  train_path, test_path = maybe_download()
  train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
  train_x, train_y = train, train.pop(y_name)
  test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
  test_x, test_y = test, test.pop(y_name)
  return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
  # convert to dataset
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
  # shuffle, repeat, and batch
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)
  return dataset

def eval_input_fn(features, labels, batch_size):
  '''an input function for evaluation or prediction'''
  features = dict(features)
  if labels is None:
    inputs = features
  else:
    inputs = (features, labels)
  dataset = tf.data.Dataset.from_tensor_slices(inputs)
  assert batch_size is not None, 'batch_size must not be None'
  dataset = dataset.batch(batch_size)
  return dataset

CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
  fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
  features = dict(zip(CSV_COLUMN_NAMES, fields))
  label = features.pop('Species')
  return features, label

def csv_input_fn(csv_path, batch_size):
  dataset = tf.data.TextLineDataset(csv_path).skip(1)
  dataset = dataset.map(_parse_line)
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)
  return dataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

def main(argv):
  args = parser.parse_args(argv[1:])
  (train_x, train_y), (test_x, test_y) = load_data()

  # feature columns describe how to use the input
  my_feature_columns = []
  for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

  # build a deep neural network
  classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    hidden_units = [10,10],
    n_classes = 3
  )

  # train the model
  classifier.train(
    input_fn = lambda: train_input_fn(train_x, train_y, args.batch_size),
    steps = args.train_steps
  )

  # evaluate the model
  eval_result = classifier.evaluate(
    input_fn = lambda: eval_input_fn(test_x, test_y, args.batch_size)
  )

  print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
