import pdb
import random
import tempfile

import cv2
import tensorflow as tf
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import fastestimator as fe
from fastestimator.dataset.data import mscoco
from fastestimator.op.numpyop.multivariate import LongestMaxSize, PadIfNeeded
from fastestimator.op.numpyop.numpyop import Delete, NumpyOp
from fastestimator.op.numpyop.univariate import ReadImage
from fastestimator.op.tensorop.loss import MeanSquaredError
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.op.tensorop.tensorop import TensorOp
from fastestimator.trace.io import BestModelSaver


def group_normalize(x, gamma, beta, num_groups=None, group_size=None, eps=1e-5):
    assert x.shape.ndims >= 2, ("Less than 2-dim Tensor passed to GroupNorm. Something's fishy.")

    num_channels = x.shape[-1]
    assert num_channels is not None, "Cannot apply GroupNorm on dynamic channels."
    assert (num_groups is None) != (group_size is None), ("You must specify exactly one of `num_groups`, `group_size`")

    if group_size is not None:
        num_groups = num_channels // group_size

    assert num_channels % num_groups == 0, ("GroupNorm: {} not divisible by {}".format(num_channels, num_groups))

    orig_shape = tf.shape(x)

    # This shape is NHWGS where G is #groups and S is group-size.
    extra_shape = [num_groups, num_channels // num_groups]
    group_shape = tf.concat([orig_shape[:-1], extra_shape], axis=-1)
    x = tf.reshape(x, group_shape)

    # The dimensions to normalize over: HWS for images, but more generally all
    # dimensions except N (batch, first) and G (cross-groups, next-to-last).
    # So more visually, normdims are the dots in N......G. (note the last one is
    # also a dot, not a full-stop, argh!)
    normdims = list(range(1, x.shape.ndims - 2)) + [x.shape.ndims - 1]
    mean, var = tf.nn.moments(x, normdims, keepdims=True)

    # Interestingly, we don't have a beta/gamma per group, but still one per
    # channel, at least according to the original paper. Reshape such that they
    # broadcast correctly.
    beta = tf.reshape(beta, extra_shape)
    gamma = tf.reshape(gamma, extra_shape)
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return tf.reshape(x, orig_shape)


class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 num_groups=None,
                 group_size=None,
                 eps=1e-5,
                 beta_init=tf.zeros_initializer(),
                 gamma_init=tf.ones_initializer(),
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        if num_groups is None and group_size is None:
            num_groups = 32

        self._num_groups = num_groups
        self._group_size = group_size
        self._eps = eps
        self._beta_init = beta_init
        self._gamma_init = gamma_init

    def build(self, input_size):
        channels = input_size[-1]
        assert channels is not None, "Cannot apply GN on dynamic channels."
        self._gamma = self.add_weight(name="gamma", shape=(channels, ), initializer=self._gamma_init, dtype=self.dtype)
        self._beta = self.add_weight(name="beta", shape=(channels, ), initializer=self._beta_init, dtype=self.dtype)
        super(GroupNormalization, self).build(input_size)

    def call(self, x):
        return group_normalize(x, self._gamma, self._beta, self._num_groups, self._group_size, self._eps)


def add_name_prefix(name, prefix=None):
    return prefix + "/" + name if prefix else name


class ReLU(tf.keras.layers.ReLU):
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


class PaddingFromKernelSize(tf.keras.layers.Layer):
    """Layer that adds padding to an image taking into a given kernel size."""
    def __init__(self, kernel_size, **kwargs):
        super(PaddingFromKernelSize, self).__init__(**kwargs)
        pad_total = kernel_size - 1
        self._pad_beg = pad_total // 2
        self._pad_end = pad_total - self._pad_beg

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = tf.TensorShape(input_shape).as_list()
        if height is not None:
            height = height + self._pad_beg + self._pad_end
        if width is not None:
            width = width + self._pad_beg + self._pad_end
        return tf.TensorShape((batch_size, height, width, channels))

    def call(self, x):
        padding = [[0, 0], [self._pad_beg, self._pad_end], [self._pad_beg, self._pad_end], [0, 0]]
        return tf.pad(x, padding)


class StandardizedConv2D(tf.keras.layers.Conv2D):
    def build(self, input_shape):
        super(StandardizedConv2D, self).build(input_shape)
        # Wrap a standardization around the conv OP.
        default_conv_op = self.convolution_op

        def standardized_conv_op(inputs, kernel):
            # Kernel has shape HWIO, normalize over HWI
            mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
            # Author code uses std + 1e-5
            return default_conv_op(inputs, (kernel - mean) / tf.sqrt(var + 1e-10))

        self.convolution_op = standardized_conv_op
        self.built = True


class BottleneckV2Unit(tf.keras.layers.Layer):
    """Implements a standard ResNet's unit (version 2).
  """
    def __init__(self, num_filters, stride=1, **kwargs):
        """Initializer.
    Args:
      num_filters: number of filters in the bottleneck.
      stride: specifies block's stride.
      **kwargs: other tf.keras.layers.Layer keyword arguments.
    """
        super(BottleneckV2Unit, self).__init__(**kwargs)
        self._num_filters = num_filters
        self._stride = stride

        self._proj = None
        self._unit_a = tf.keras.Sequential([
            GroupNormalization(name="group_norm"),
            ReLU(),
        ], name="a")
        self._unit_a_conv = StandardizedConv2D(filters=num_filters,
                                               kernel_size=1,
                                               use_bias=False,
                                               padding="VALID",
                                               trainable=self.trainable,
                                               name="a/standardized_conv2d")

        self._unit_b = tf.keras.Sequential([
            GroupNormalization(name="group_norm"),
            ReLU(),
            PaddingFromKernelSize(kernel_size=3),
            StandardizedConv2D(filters=num_filters,
                               kernel_size=3,
                               strides=stride,
                               use_bias=False,
                               padding="VALID",
                               trainable=self.trainable,
                               name="standardized_conv2d")
        ],
                                           name="b")

        self._unit_c = tf.keras.Sequential([
            GroupNormalization(name="group_norm"),
            ReLU(),
            StandardizedConv2D(filters=4 * num_filters,
                               kernel_size=1,
                               use_bias=False,
                               padding="VALID",
                               trainable=self.trainable,
                               name="standardized_conv2d")
        ],
                                           name="c")

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()

        # Add projection layer if necessary.
        if (self._stride > 1) or (4 * self._num_filters != input_shape[-1]):
            self._proj = StandardizedConv2D(filters=4 * self._num_filters,
                                            kernel_size=1,
                                            strides=self._stride,
                                            use_bias=False,
                                            padding="VALID",
                                            trainable=self.trainable,
                                            name="a/proj/standardized_conv2d")
        self.built = True

    def compute_output_shape(self, input_shape):
        current_shape = self._unit_a.compute_output_shape(input_shape)
        current_shape = self._unit_a_conv.compute_output_shape(current_shape)
        current_shape = self._unit_b.compute_output_shape(current_shape)
        current_shape = self._unit_c.compute_output_shape(current_shape)
        return current_shape

    def call(self, x):
        x_shortcut = x
        # Unit "a".
        x = self._unit_a(x)
        if self._proj is not None:
            x_shortcut = self._proj(x)
        x = self._unit_a_conv(x)
        # Unit "b".
        x = self._unit_b(x)
        # Unit "c".
        x = self._unit_c(x)

        return x + x_shortcut


class ResnetV2(tf.keras.Model):
    """Generic ResnetV2 architecture, as used in the BiT paper."""
    def __init__(self, num_units=(3, 4, 6, 3), num_outputs=1000, filters_factor=4, strides=(1, 2, 2, 2), **kwargs):
        super(ResnetV2, self).__init__(**kwargs)

        num_blocks = len(num_units)
        num_filters = tuple(16 * filters_factor * 2**b for b in range(num_blocks))

        self._root = self._create_root_block(num_filters=num_filters[0])
        self._blocks = []
        for b, (f, u, s) in enumerate(zip(num_filters, num_units, strides), 1):
            n = "block{}".format(b)
            self._blocks.append(self._create_block(num_units=u, num_filters=f, stride=s, name=n))
        self._pre_head = [GroupNormalization(name="group_norm"), ReLU(), tf.keras.layers.GlobalAveragePooling2D()]
        self._head = None
        if num_outputs:
            self._head = tf.keras.layers.Dense(units=num_outputs,
                                               use_bias=True,
                                               kernel_initializer="zeros",
                                               trainable=self.trainable,
                                               name="head/dense")

    def _create_root_block(self, num_filters, conv_size=7, conv_stride=2, pool_size=3, pool_stride=2):
        layers = [
            PaddingFromKernelSize(conv_size),
            StandardizedConv2D(filters=num_filters,
                               kernel_size=conv_size,
                               strides=conv_stride,
                               trainable=self.trainable,
                               use_bias=False,
                               name="standardized_conv2d"),
            PaddingFromKernelSize(pool_size),
            tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_stride, padding="valid")
        ]
        return tf.keras.Sequential(layers, name="root_block")

    def _create_block(self, num_units, num_filters, stride, name):
        layers = []
        for i in range(1, num_units + 1):
            layers.append(
                BottleneckV2Unit(num_filters=num_filters, stride=(stride if i == 1 else 1), name="unit%02d" % i))
        return tf.keras.Sequential(layers, name=name)

    def compute_output_shape(self, input_shape):
        current_shape = self._root.compute_output_shape(input_shape)
        for block in self._blocks:
            current_shape = block.compute_output_shape(current_shape)
        for layer in self._pre_head:
            current_shape = layer.compute_output_shape(current_shape)
        if self._head is not None:
            batch_size, features = current_shape.as_list()
            current_shape = (batch_size, 1, 1, features)
            current_shape = self._head.compute_output_shape(current_shape).as_list()
            current_shape = (current_shape[0], current_shape[3])
        return tf.TensorShape(current_shape)

    def call(self, x):
        x = self._root(x)
        for block in self._blocks:
            x = block(x)
        for layer in self._pre_head:
            x = layer(x)
        if self._head is not None:
            x = self._head(x)
        return x


class SentenceEmbedding(NumpyOp):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, data, state):
        caption_list = data
        # Randomly select one caption if there are multiple captions
        selected_caption = random.choice(caption_list)
        # Tokenize sentences
        encoded_input = self.tokenizer(selected_caption, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
            # Perform pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0]


def resnetv2_bit(input_shape=(None, None, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    r50_bit = ResnetV2(num_outputs=21843)
    r50_bit.build((None, None, None, 3))
    r50_bit.load_weights("bit_r50.h5")
    x = inputs
    for layer in r50_bit.layers[:-1]:
        x = layer(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


class Rescale(TensorOp):
    def forward(self, data, state):
        return (tf.cast(data, tf.float32) - 127.5) / 127.5


def encoding_mapping(bit_dim=2048, text_dim=384):
    inputs = tf.keras.layers.Input(shape=(bit_dim, ))
    x = tf.keras.layers.Dense(text_dim)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=x)


def get_estimator(save_dir=tempfile.mkdtemp()):
    train_ds, eval_ds = mscoco.load_data("/raid/shared_data/", load_bboxes=False, load_captions=True, replacement=False)
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=eval_ds,
        batch_size=64,
        ops=[
            ReadImage(inputs="image", outputs="image"),
            LongestMaxSize(max_size=256, image_in="image"),
            PadIfNeeded(min_height=256, min_width=256, image_in="image", border_mode=cv2.BORDER_CONSTANT, value=0),
            SentenceEmbedding(inputs="caption", outputs="text_encoding"),
            Delete(keys=("image_id", "caption"))
        ])
    # tf.io.gfile.copy("gs://bit_models/BiT-M-R50x1.h5", "bit_r50.h5")
    bit_encoder = fe.build(resnetv2_bit, optimizer_fn=None)
    encoding_map = fe.build(encoding_mapping, optimizer_fn="sgd")
    network = fe.Network(ops=[
        Rescale(inputs="image", outputs="image"),
        ModelOp(model=bit_encoder, inputs="image", outputs="image_encoding", trainable=False),
        ModelOp(model=encoding_map, inputs="image_encoding", outputs="text_encoding_pred"),
        MeanSquaredError(inputs=("text_encoding_pred", "text_encoding"), outputs="mse"),
        UpdateOp(model=encoding_map, loss_name="mse")
    ])
    traces = [BestModelSaver(model=encoding_map, save_dir=save_dir)]
    esitmator = fe.Estimator(pipeline=pipeline, network=network, epochs=100, traces=traces)
    return esitmator