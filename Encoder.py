import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data("MNIST_data/",one_hot=True)
# Parameters
learning_rate = 0.01  # 0.01 this learning rate will be better! Tested
training_epochs = 10  # 10 Epoch 训练
batch_size = 256
display_step = 1
# hidden layer settings
n_input = 784
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], )),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], )),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], )),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], )),

    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3], )),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2], )),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], )),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], )),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                     biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    return layer_4


# Construct model
encoder_op = encoder(X)  # 128 Features
decoder_op = decoder(encoder_op)  # 784 Features

# Prediction
y_pred = decoder_op  # After
# Targets (Labels) are the input data.
y_true = X  # Before

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# Launch the graph
with tf.Session() as sess:
    # tf 马上就要废弃tf.initialize_all_variables()这种写法
    # 替换成下面:
    sess.run(tf.global_variables_initializer())
    total_batch = int(mnist.train.num_examples / batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()
