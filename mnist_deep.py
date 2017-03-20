#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-2-24 下午11:09
# @Author  : ywl
# @File    : mnist_deep.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义变量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

y_pred = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))

# 训练模型
sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
    batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

correct_prediction = tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})


# print sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels})
print

# 随机初始化权重
def weight_variable(shape):
    return tf.Variable(tf.truncate_normal(shape, stddev=0.1))


# 随机初始化Sbiases
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


print(1)
