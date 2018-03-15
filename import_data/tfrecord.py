import tensorflow as tf
from PIL import Image
import numpy as np
import os

'''
  write tfrecords
'''
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecord(img_name, tf_name):
  filename = os.path.join('.', tf_name)
  with tf.python_io.TFRecordWriter(filename) as writer:
    img = np.array(Image.open(os.path.join('.', img_name)))
    height = img.shape[0]
    width = img.shape[1]
    img_raw = img.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(height),
      'width': _int64_feature(width),
      'image_raw': _bytes_feature(img_raw)
    }))
    writer.write(example.SerializeToString())

write_tfrecord('1.jpg', 'f1.tfrecords')
write_tfrecord('2.jpg', 'f2.tfrecords')

'''
  parse tfrecords(images)
'''
def _parse(example_proto):
  features = {
    'image': tf.FixedLenFeature((), tf.string, default_value=''),
    'label': tf.FixedLenFeature((), tf.int64, default_value=0)
  }
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features['image'], parsed_features['label']

filenames = tf.constant([os.path.join('.', 'f1.tfrecords'), os.path.join('.', 'f2.tfrecords')])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse)
