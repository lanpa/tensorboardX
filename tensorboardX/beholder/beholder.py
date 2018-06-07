# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ..src.summary_pb2 import Summary
from ..src.summary_pb2 import SummaryMetadata
from ..src.tensor_pb2 import TensorProto
from ..src.tensor_shape_pb2 import TensorShapeProto

import os
import time

import numpy as np
import tensorflow as tf

# from tensorboard.plugins.beholder import im_util
# from . import im_util
from .file_system_tools import read_pickle,\
  write_pickle, write_file
from .shared_config import PLUGIN_NAME, TAG_NAME,\
  SUMMARY_FILENAME, DEFAULT_CONFIG, CONFIG_FILENAME, SUMMARY_COLLECTION_KEY_NAME
from . import video_writing
# from .visualizer import Visualizer

# exit()

class Beholder(object):

  def __init__(self, logdir):
    self.PLUGIN_LOGDIR = logdir + '/plugins/' + PLUGIN_NAME

    self.is_recording = False
    self.video_writer = video_writing.VideoWriter(
        self.PLUGIN_LOGDIR,
        outputs=[
            video_writing.FFmpegVideoOutput,
            video_writing.PNGVideoOutput])

    self.frame_placeholder = tf.placeholder(tf.uint8, [None, None, None])
    self.summary_op = tf.summary.tensor_summary(TAG_NAME,
                                                self.frame_placeholder,
                                                collections=[
                                                    SUMMARY_COLLECTION_KEY_NAME
                                                ])

    self.last_image_shape = []
    self.last_update_time = time.time()
    self.config_last_modified_time = -1
    self.previous_config = dict(DEFAULT_CONFIG)

    if not os.path.exists(self.PLUGIN_LOGDIR + '/config.pkl'):
      os.makedirs(self.PLUGIN_LOGDIR)
      write_pickle(DEFAULT_CONFIG, '{}/{}'.format(self.PLUGIN_LOGDIR,
                                                  CONFIG_FILENAME))

    # self.visualizer = Visualizer(self.PLUGIN_LOGDIR)


  def _get_config(self):
    '''Reads the config file from disk or creates a new one.'''
    filename = '{}/{}'.format(self.PLUGIN_LOGDIR, CONFIG_FILENAME)
    modified_time = os.path.getmtime(filename)

    if modified_time != self.config_last_modified_time:
      config = read_pickle(filename, default=self.previous_config)
      self.previous_config = config
    else:
      config = self.previous_config

    self.config_last_modified_time = modified_time
    return config


  def _write_summary(self, frame):
    '''Writes the frame to disk as a tensor summary.'''
    
    # summary = session.run(self.summary_op, feed_dict={
    #     self.frame_placeholder: frame
    # })
    path = '{}/{}'.format(self.PLUGIN_LOGDIR, SUMMARY_FILENAME)
    # write_file(summary, path)

    # PluginData = [SummaryMetadata.PluginData(plugin_name=TAG_NAME)]
    data = np.random.randn(8, 8)
    smd = SummaryMetadata()
    tensor = TensorProto(dtype='DT_FLOAT',
                         float_val=data.reshape(-1).tolist(),
                         tensor_shape=TensorShapeProto(
                             dim=[TensorShapeProto.Dim(size=data.shape[0]), TensorShapeProto.Dim(size=data.shape[1])]))
    summary = Summary(value=[Summary.Value(tag=TAG_NAME, metadata=smd, tensor=tensor)]).SerializeToString()
    write_file(summary, path)
    

  def _get_final_image(self, config, trainable=None, arrays=None, frame=None):
    if config['values'] == 'frames':
      print('===frames===')
      # if frame is None:
      #   final_image = im_util.get_image_relative_to_script('frame-missing.png')
      # else:
      #   frame = frame() if callable(frame) else frame
      #   final_image = im_util.scale_image_for_display(frame)
      final_image = frame

    elif config['values'] == 'arrays':
      print('===arrays===')
      # if arrays is None:
      #   final_image = im_util.get_image_relative_to_script('arrays-missing.png')
      #   # TODO: hack to clear the info. Should be cleaner.
      #   self.visualizer._save_section_info([], [])
      # else:
      # final_image = self.visualizer.build_frame(arrays)
      final_image = np.random.randn(256, 600)
    elif config['values'] == 'trainable_variables':
      print('===trainable===')
      # arrays = [session.run(x) for x in tf.trainable_variables()]
      # final_image = self.visualizer.build_frame(trainable)
      final_image = np.random.randn(128, 600)
    if len(final_image.shape) == 2:
      # Map grayscale images to 3D tensors.
      final_image = np.expand_dims(final_image, -1)

    return final_image


  def _enough_time_has_passed(self, FPS):
    '''For limiting how often frames are computed.'''
    if FPS == 0:
      return False
    else:
      earliest_time = self.last_update_time + (1.0 / FPS)
      return time.time() >= earliest_time


  def _update_frame(self, trainable, arrays, frame, config):
    final_image = self._get_final_image(config, trainable, arrays, frame)
    self._write_summary(final_image)
    self.last_image_shape = final_image.shape

    return final_image


  def _update_recording(self, frame, config):
    '''Adds a frame to the current video output.'''
    # pylint: disable=redefined-variable-type
    should_record = config['is_recording']

    if should_record:
      if not self.is_recording:
        self.is_recording = True
        print(
            'Starting recording using %s',
            self.video_writer.current_output().name())
      self.video_writer.write_frame(frame)
    elif self.is_recording:
      self.is_recording = False
      self.video_writer.finish()
      print('Finished recording')


  # TODO: blanket try and except for production? I don't someone's script to die
  #       after weeks of running because of a visualization.
  def update(self, session, trainable=None, arrays=None, frame=None):
    '''Creates a frame and writes it to disk.

    Args:
      arrays: a list of np arrays. Use the "custom" option in the client.
      frame: a 2D np array. This way the plugin can be used for video of any
             kind, not just the visualization that comes with the plugin.

             frame can also be a function, which only is evaluated when the
             "frame" option is selected by the client.
    '''
    new_config = self._get_config()

    if self._enough_time_has_passed(self.previous_config['FPS']):
      # self.visualizer.update(new_config)
      self.last_update_time = time.time()
      final_image = self._update_frame(trainable, arrays, frame, new_config)
      self._update_recording(final_image, new_config)


  ##############################################################################

  @staticmethod
  def gradient_helper(optimizer, loss, var_list=None):
    '''A helper to get the gradients out at each step.

    Args:
      optimizer: the optimizer op.
      loss: the op that computes your loss value.

    Returns: the gradient tensors and the train_step op.
    '''
    if var_list is None:
      var_list = tf.trainable_variables()

    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    grads = [pair[0] for pair in grads_and_vars]

    return grads, optimizer.apply_gradients(grads_and_vars)




# implements pytorch backward later
class BeholderHook():
  pass
  # """SessionRunHook implementation that runs Beholder every step.

  # Convenient when using tf.train.MonitoredSession:
  # ```python
  # beholder_hook = BeholderHook(LOG_DIRECTORY)
  # with MonitoredSession(..., hooks=[beholder_hook]) as sess:
  #   sess.run(train_op)
  # ```
  # """
  # def __init__(self, logdir):
  #   """Creates new Hook instance

  #   Args:
  #     logdir: Directory where Beholder should write data.
  #   """
  #   self._logdir = logdir
  #   self.beholder = None

  # def begin(self):
  #   self.beholder = Beholder(self._logdir)

  # def after_run(self, run_context, unused_run_values):
  #   self.beholder.update(run_context.session)
