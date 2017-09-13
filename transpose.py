import numpy as np
import tensorflow as tf

x = np.arange(8*8*8).reshape([8,8,8])

with tf.device('/gpu:0'):
	sess = tf.Session()
	m = tf.constant(x)
	m = tf.transpose(m, perm=[0, 2, 1])
	res = sess.run(m)
print res