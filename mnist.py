# -*- coding: utf-8 -*-
#
# Created by PyCharm
# Date: 16-6-30
# Time: 上午10:15
# Author: Zhu Danxiang
#

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 784])
# x为实际的输入,None表示大小不确定,shape为可选参数,带上tensorflow可以捕捉到维度不一致的错误

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
# y为预测的输出

y_ = tf.placeholder(tf.float32, [None, 10])
# y_为实际的输出

# 计算交叉熵代价函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

print sess.run(tf.arg_max(y, 1), feed_dict={x: mnist.test.images[1:2], y_: mnist.test.labels[1:2]})
sess.close()
