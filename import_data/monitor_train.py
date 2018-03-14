import tensorflow as tf

dataset = tf.data.Dataset.range(100)
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.train.MonitoredTrainingSession() as sess:
  while not sess.should_stop():
    print(sess.run(next_element)),

with tf.train.MonitoredTrainingSession() as sess:
  for _ in range(101):
    print(sess.run(next_element)),
