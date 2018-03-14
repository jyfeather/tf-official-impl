import tensorflow as tf

'''
  create a mini-batch
'''
dataset = tf.data.Dataset.range(100)
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
  print(sess.run(next_element))

'''
  padding batch
'''
dataset = tf.data.Dataset.range(13)

dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))

dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
  print(sess.run(next_element))
  print(sess.run(next_element))
