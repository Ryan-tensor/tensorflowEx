# Lab 12 RNN
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

idx2char = ['c', 'o', 'f', 'e', ' ', 'a']
# Teach hello: coffee coc -> offee coca
x_data = [[0, 1, 2, 2, 3, 3, 4, 0, 1, 0]]   # coffe coca
x_one_hot = [[[1, 0, 0, 0, 0, 0],   # c 0
              [0, 1, 0, 0, 0, 0],   # o 1
              [0, 0, 1, 0, 0, 0],   # f 2
              [0, 0, 1, 0, 0, 0],   # f 2
              [0, 0, 0, 1, 0, 0],   # e 3
              [0, 0, 0, 1, 0, 0],   # e 3
              [0, 0, 0, 0, 1, 0],   # ' ' 4
              [1, 0, 0, 1, 0, 0],   # c 0
              [0, 1, 0, 1, 0, 0],   # o 1
              [1, 0, 0, 1, 0, 0],   # c 0
              ]]

y_data = [[1, 2, 2, 3, 3, 4, 0, 1, 0, 5]]    # offee coca

num_classes = 6
input_dim = 6  # one-hot size
hidden_size = 6  # output from the LSTM. 6 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 10  # |offee coca| == 10
learning_rate = 0.1

X = tf.placeholder(
    tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))