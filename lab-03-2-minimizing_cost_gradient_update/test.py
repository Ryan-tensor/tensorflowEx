import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y)) ##

learning_rate = 0.1
gradient = tf.reduce_mean((hypothesis - Y) * X)   ##
descent = W - learning_rate * gradient            ##
update = W.assign(descent)                        ##

sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))