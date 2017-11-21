"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np
import time

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

fil_depth = 16
fil_size = 3
fcn_units = 128

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Introduce variables for easy changing of filter depths
  #fil_depth = 16
  #fil_size = 3
  #fcn_units = 128
  print('')

  # Reshape to use within a convolutional neural net.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([fil_size, fil_size, 1, fil_depth])
    b_conv1 = bias_variable([fil_depth])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([fil_size, fil_size, fil_depth, fil_depth])
    b_conv2 = bias_variable([fil_depth])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 128 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * fil_depth, fcn_units])
    b_fc1 = bias_variable([fcn_units])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*fil_depth])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 128 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([fcn_units, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def plot_record(trainAcc):
  """plots out training accuracy"""
  plt.plot(trainAcc, 'b-', label='Training Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Accuracy of Network')
  plt.show()


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('grad_desc_optimizer'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    max_epochs = 20
    epoch_train_acc = []

    t1 = time.time()
    for e in range(max_epochs):
      print('')
      print('Current epoch: %d' % e)
      batch_train_acc = []
      min_batch_train_acc = []
      for i in range(1000):
        batch = mnist.train.next_batch(64)
        if i % 10 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
          min_batch_train_acc.append(train_accuracy)
          # print('Step: %d, Training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        batch_train_acc.append((sum(min_batch_train_acc)/len(min_batch_train_acc))) # 100

      print('Training accuracy %g' % (train_accuracy))
      epoch_train_acc.append((sum(batch_train_acc)/len(batch_train_acc)))

    parameters = tf.trainable_variables()
    t2 = time.time()
    delta_t = t2 - t1
    total_parameters = 0
    for variable in tf.trainable_variables():
      # shape is an array of tf.Dimension
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      total_parameters += variable_parameters

    print('Number of filters: %d' % fil_depth)
    print('Time taken: %f' % delta_t)
    print('Parameters: ', total_parameters)
    print('---------------------------------------')
    print('Test accuracy: %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    plot_record(epoch_train_acc)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
