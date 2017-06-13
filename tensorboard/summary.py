# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""## Generation of summaries.
### Class for writing Summaries
@@FileWriter
@@FileWriterCache
### Summary Ops
@@tensor_summary
@@scalar
@@histogram
@@audio
@@image
@@merge
@@merge_all
## Utilities
@@get_summary_description
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import re as _re
import bisect
from six import StringIO
from six.moves import range
from PIL import Image
import numpy as np

# pylint: disable=unused-import
from .src.summary_pb2 import Summary
from .src.summary_pb2 import HistogramProto


_INVALID_TAG_CHARACTERS = _re.compile(r'[^-/\w\.]')


def _clean_tag(name):
  # In the past, the first argument to summary ops was a tag, which allowed
  # arbitrary characters. Now we are changing the first argument to be the node
  # name. This has a number of advantages (users of summary ops now can
  # take advantage of the tf name scope system) but risks breaking existing
  # usage, because a much smaller set of characters are allowed in node names.
  # This function replaces all illegal characters with _s, and logs a warning.
  # It also strips leading slashes from the name.
  if name is not None:
    new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
    new_name = new_name.lstrip('/')  # Remove leading slashes
    if new_name != name:
      logging.info(
          'Summary name %s is illegal; using %s instead.' %
          (name, new_name))
      name = new_name
  return name


def scalar(name, scalar, collections=None):
    """Outputs a `Summary` protocol buffer containing a single scalar value.
    The generated Summary has a Tensor.proto containing the input Tensor.
    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: A real numeric Tensor containing a single value.
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    Returns:
      A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
    Raises:
      ValueError: If tensor has the wrong shape or type.
    """
    name = _clean_tag(name)
    if not isinstance(scalar, float):
        # try conversion, if failed then need handle by user.
        scalar = float(scalar)
    return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])


def histogram(name, values, collections=None):
    # pylint: disable=line-too-long
    """Outputs a `Summary` protocol buffer with a histogram.
    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.
    This op reports an `InvalidArgument` error if any value is not finite.
    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      values: A real numeric `Tensor`. Any shape. Values to use to
        build the histogram.
      collections: Optional list of graph collections keys. The new summary op is
        added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    name = _clean_tag(name)
    hist = make_histogram(values.astype(float))
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def make_histogram_buckets():
    v = 1E-12
    buckets = []
    neg_buckets = []
    while v < 1E20:
        buckets.append(v)
        neg_buckets.append(-v)
        v *= 1.1
    # Should include DBL_MAX, but won't bother for test data.
    return neg_buckets[::-1] + [0] + buckets


def make_histogram(values):
    """Convert values into a histogram proto using logic from histogram.cc."""
    limits = make_histogram_buckets()
    counts = [0] * len(limits)
    for v in values:
        idx = bisect.bisect_left(limits, v)
        counts[idx] += 1

    limit_counts = [(limits[i], counts[i]) for i in range(len(limits))
                   if counts[i]]
    bucket_limit = [lc[0] for lc in limit_counts]
    bucket = [lc[1] for lc in limit_counts]
    sum_sq = sum(v * v for v in values)
    return HistogramProto(min=min(values),
                          max=max(values),
                          num=len(values),
                          sum=sum(values),
                          sum_squares=sum_sq,
                          bucket_limit=bucket_limit,
                          bucket=bucket)


def image(tag, tensor):
    """Outputs a `Summary` protocol buffer with images.
    The summary has up to `max_images` summary values containing images. The
    images are built from `tensor` which must be 3-D with shape `[height, width,
    channels]` and where `channels` can be:
    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.
    The `name` in the outputted Summary.Value protobufs is generated based on the
    name, with a suffix depending on the max_outputs setting:
    *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
    *  If `max_outputs` is greater than 1, the summary value tags are
       generated sequentially as '*name*/image/0', '*name*/image/1', etc.
    Args:
      tag: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `uint8` or `float32` `Tensor` of shape `[height, width,
        channels]` where `channels` is 1, 3, or 4.
    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer.
    """
    tag = _clean_tag(tag)
    if not isinstance(tensor, np.ndarray):
        # try conversion, if failed then need handle by user.
        tensor = np.ndarray(tensor, dtype=np.float32)
    shape = tensor.shape
    height, width, channel = shape[0], shape[1], shape[2]
    if channel == 1:
        # walk around. PIL's setting on dimension.
        tensor = np.reshape(tensor, (height, width))
    image = make_image(tensor, height, width, channel)
    return Summary(value=[Summary.Value(tag=tag, image=image)])


def make_image(tensor, height, width, channel):
    """Convert an numpy representation image to Image protobuf"""
    image = Image.fromarray(tensor)
    #output = StringIO()
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)



'''TODO(zihaolucky). support more summary types later.
def audio(name, tensor, sample_rate, max_outputs=3, collections=None):
  # pylint: disable=line-too-long
  """Outputs a `Summary` protocol buffer with audio.
  The summary has up to `max_outputs` summary values containing audio. The
  audio is built from `tensor` which must be 3-D with shape `[batch_size,
  frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
  assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
  `sample_rate`.
  The `tag` in the outputted Summary.Value protobufs is generated based on the
  name, with a suffix depending on the max_outputs setting:
  *  If `max_outputs` is 1, the summary value tag is '*name*/audio'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*name*/audio/0', '*name*/audio/1', etc
  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    tensor: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
      or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
    sample_rate: A Scalar `float32` `Tensor` indicating the sample rate of the
      signal in hertz.
    max_outputs: Max number of batch elements to generate audio for.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]
  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  """
  # pylint: enable=line-too-long
  name = _clean_tag(name)
  with _ops.name_scope(name, None, [tensor]) as scope:
    # pylint: disable=protected-access
    sample_rate = _ops.convert_to_tensor(
        sample_rate, dtype=_dtypes.float32, name='sample_rate')
    val = _gen_logging_ops._audio_summary_v2(
        tag=scope.rstrip('/'),
        tensor=tensor,
        max_outputs=max_outputs,
        sample_rate=sample_rate,
        name=scope)
    _collect(val, collections, [_ops.GraphKeys.SUMMARIES])
  return val


def merge(inputs, collections=None, name=None):
  # pylint: disable=line-too-long
  """Merges summaries.
  This op creates a
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  protocol buffer that contains the union of all the values in the input
  summaries.
  When the Op is run, it reports an `InvalidArgument` error if multiple values
  in the summaries to merge use the same tag.
  Args:
    inputs: A list of `string` `Tensor` objects containing serialized `Summary`
      protocol buffers.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    name: A name for the operation (optional).
  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer resulting from the merging.
  """
  # pylint: enable=line-too-long
  name = _clean_tag(name)
  with _ops.name_scope(name, 'Merge', inputs):
    # pylint: disable=protected-access
    val = _gen_logging_ops._merge_summary(inputs=inputs, name=name)
    _collect(val, collections, [])
  return val


def merge_all(key=_ops.GraphKeys.SUMMARIES):
  """Merges all summaries collected in the default graph.
  Args:
    key: `GraphKey` used to collect the summaries.  Defaults to
      `GraphKeys.SUMMARIES`.
  Returns:
    If no summaries were collected, returns None.  Otherwise returns a scalar
    `Tensor` of type `string` containing the serialized `Summary` protocol
    buffer resulting from the merging.
  """
  summary_ops = _ops.get_collection(key)
  if not summary_ops:
    return None
  else:
    return merge(summary_ops)


def get_summary_description(node_def):
  """Given a TensorSummary node_def, retrieve its SummaryDescription.
  When a Summary op is instantiated, a SummaryDescription of associated
  metadata is stored in its NodeDef. This method retrieves the description.
  Args:
    node_def: the node_def_pb2.NodeDef of a TensorSummary op
  Returns:
    a summary_pb2.SummaryDescription
  Raises:
    ValueError: if the node is not a summary op.
  """

  if node_def.op != 'TensorSummary':
    raise ValueError("Can't get_summary_description on %s" % node_def.op)
  description_str = _compat.as_str_any(node_def.attr['description'].s)
  summary_description = SummaryDescription()
  _json_format.Parse(description_str, summary_description)
  return summary_description
'''
