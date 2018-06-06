# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple MNIST classifier to demonstrate features of Beholder.

Based on tensorflow/examples/tutorials/mnist/mnist_with_summaries.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import torch

import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist
import tensorboardX.beholder as beholder_lib

FLAGS = None

LOG_DIRECTORY = '/tmp/beholder-demo'

def train():
  mnist_data = mnist.input_data.read_data_sets(
      FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()

  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  #conv1
  kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 10],
                                           dtype=tf.float32,
                                           stddev=1e-1),
                       name='conv-weights')
  conv = tf.nn.conv2d(image_shaped_input, kernel, [1, 1, 1, 1], padding='VALID')
  biases_init = tf.constant(
      0.0, shape=[kernel.get_shape().as_list()[-1]], dtype=tf.float32)
  biases = tf.Variable(biases_init, trainable=True, name='biases')
  out = tf.nn.bias_add(conv, biases)
  conv1 = tf.nn.relu(out, name='relu')

  #conv2
  kernel2_init = tf.truncated_normal(
      [3, 3, 10, 20], dtype=tf.float32, stddev=1e-1)
  kernel2 = tf.Variable(kernel2_init, name='conv-weights2')
  conv2 = tf.nn.conv2d(conv1, kernel2, [1, 1, 1, 1], padding='VALID')
  biases2_init = tf.constant(
      0.0, shape=[kernel2.get_shape().as_list()[-1]], dtype=tf.float32)
  biases2 = tf.Variable(biases2_init, trainable=True, name='biases')
  out2 = tf.nn.bias_add(conv2, biases2)
  conv2 = tf.nn.relu(out2, name='relu')

  flattened = tf.contrib.layers.flatten(conv2)
  hidden1 = nn_layer(
      flattened, flattened.get_shape().as_list()[1], 10, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  y = nn_layer(dropped, 10, 10, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    gradients, train_step = beholder_lib.Beholder.gradient_helper(
        optimizer, cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(LOG_DIRECTORY + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(LOG_DIRECTORY + '/test')
  tf.global_variables_initializer().run()

  beholder = beholder_lib.Beholder(logdir=LOG_DIRECTORY)


  def feed_dict(is_train):
    if is_train or FLAGS.fake_data:
      xs, ys = mnist_data.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist_data.test.images, mnist_data.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
    print('i', i)
    feed_dictionary = feed_dict(True)
    summary, gradient_arrays, activations, _ = sess.run(
        [
            merged,
            gradients,
            [image_shaped_input, conv1, conv2, hidden1, y],
            train_step
        ],
        feed_dict=feed_dictionary)
    fake_param = [torch.rand(128, 768).numpy() for i in range(5)]
    arrays = [torch.rand(128, 768).numpy() for i in range(10)]
    train_writer.add_summary(summary, i)
    first_of_batch = sess.run(x, feed_dict=feed_dictionary)[0].reshape(28, 28)
    beholder.update(
        session=sess,
        trainable=fake_param,
        arrays=arrays,
        frame=torch.rand(40, 40).numpy()*255,
    )

  train_writer.close()
  test_writer.close()


def beholder_pytorch():
  arrays = [torch.rand(128, 768) for i in range(10)]
  beholder = beholder_lib.Beholder(logdir=LOG_DIRECTORY)
  beholder.update(
    session=sess,
    arrays=arrays,
    frame=torch.rand(40, 40),
)


def main(_):
  import os
  if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)
  print(LOG_DIRECTORY)
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=10000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
