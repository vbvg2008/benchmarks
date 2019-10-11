import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

fmap_base = 8192  # Overall multiplier for the number of feature maps.
fmap_decay = 1.0  # log2 feature map reduction when doubling the resolution.
fmap_max = 512  # Maximum number of feature maps in any layer.


def nf(stage):
    return min(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_max)


class FadeIn(layers.Add):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _merge_function(self, inputs):
        assert len(inputs) == 2, "FadeIn only supports two layers"
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


class PixelNormalization(layers.Layer):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def call(self, input):
        return input * tf.math.rsqrt(tf.reduce_sum(tf.square(input), axis=-1, keepdims=True) + self.eps)


class MiniBatchStd(layers.Layer):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def call(self, input):
        group_size = tf.minimum(self.group_size, tf.shape(input)[0])
        s = input.shape  # [NHWC]
        y = tf.reshape(input, [group_size, -1, s[1], s[2], s[3]])  # [GMHWC]
        y = tf.cast(y, tf.float32)  # [GMHWC]
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMHWC]
        y = tf.reduce_mean(tf.square(y), axis=0)  #[MHWC]
        y = tf.sqrt(y + 1e-8)  # [MHWC]
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]
        y = tf.cast(y, input.dtype)  # [M111]
        y = tf.tile(y, [self.group_size, s[1], s[2], 1])  # [NHW1]
        return tf.concat([input, y], axis=-1)


class EqualizedLRDense(layers.Layer):
    def __init__(self, units, gain=np.sqrt(2)):
        super().__init__()
        self.units = units
        self.gain = gain

    def build(self, input_shape):
        self.w = self.add_weight(shape=[int(input_shape[-1]), self.units],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
                                 trainable=True)
        fan_in = np.prod(input_shape[-1])
        self.wscale = tf.constant(np.float32(self.gain / np.sqrt(fan_in)))

    def call(self, input):
        return tf.matmul(input, self.w) * self.wscale


class EqualizedLRConv2D(layers.Conv2D):
    def __init__(self, filters, gain=np.sqrt(2), kernel_size=3, strides=(1, 1), padding="same"):
        super().__init__(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         use_bias=False,
                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        self.gain = gain

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            if input_shape.dims[channel_axis].value is None:
                raise ValueError('The channel dimension of the inputs ' 'should be defined. Found `None`.')
        super().build(input_shape)
        input_dim = int(input_shape[channel_axis])
        fan_in = np.prod(input_shape[1:])
        self.wscale = tf.constant(np.float32(self.gain / np.sqrt(fan_in)))

    def call(self, input):
        return super().call(input) * self.wscale


class ApplyBias(layers.Layer):
    def build(self, input_shape):
        self.b = self.add_weight(shape=input_shape[-1], initializer='zeros', trainable=True)

    def call(self, input):
        # \NOTE(JP): The original code uses "tied" bias.
        if len(input.shape) == 2:
            return input + self.b
        else:
            return input + tf.reshape(self.b, [1, 1, 1, -1])


def block_G(res, latent_dim=512, num_channels=3, target_res=10):
    if res == 2:
        x0 = layers.Input(shape=(latent_dim, ))
        x = PixelNormalization()(x0)

        #         x = layers.Dense(units=nf(res - 1) * 16)(x)
        x = EqualizedLRDense(units=nf(res - 1) * 16, gain=np.sqrt(2) / 4)(x)
        #         x = ApplyBias()(x)
        x = tf.reshape(x, [-1, 4, 4, nf(res - 1)])
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = PixelNormalization()(x)

        #         x = layers.Conv2D(filters=nf(res - 1), kernel_size=3, padding="same")(x)
        x = EqualizedLRConv2D(filters=nf(res - 1))(x)
        x = ApplyBias()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = PixelNormalization()(x)
    else:
        x0 = layers.Input(shape=(2**(res - 1), 2**(res - 1), nf(res - 2)))
        x = layers.UpSampling2D()(x0)
        for _ in range(2):
            #             x = layers.Conv2D(filters=nf(res - 1), kernel_size=3, padding="same")(x)
            x = EqualizedLRConv2D(filters=nf(res - 1))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = PixelNormalization()(x)
    return Model(inputs=x0, outputs=x, name="g_block_%dx%d" % (2**res, 2**res))


def torgb(res, num_channels=3):  # res = 2..resolution_log2
    x0 = layers.Input(shape=(2**res, 2**res, nf(res - 1)))
    #     x = layers.Conv2D(filters=num_channels, kernel_size=1, padding="same")(x0)
    x = EqualizedLRConv2D(filters=num_channels, kernel_size=1, gain=1.0)(x0)
    x = ApplyBias()(x)
    return Model(inputs=x0, outputs=x, name="to_rgb_%dx%d" % (2**res, 2**res))


def build_G(latent_dim=512, initial_resolution=2, target_resolution=10):
    x0 = layers.Input(shape=(latent_dim, ))
    curr_g_block = block_G(initial_resolution)
    curr_to_rgb_block = torgb(initial_resolution)
    images_out = curr_g_block(x0)
    images_out = curr_to_rgb_block(images_out)
    alpha = tf.Variable(initial_value=1.0, dtype='float32', trainable=False)

    model_list = list()
    gen_block_list = list()

    mdl = Model(inputs=x0, outputs=images_out)
    mdl.alpha = alpha
    model_list.append(mdl)

    gen_block_list.append(curr_g_block)

    prev_g_block = curr_g_block
    prev_to_rgb_block = curr_to_rgb_block

    for res in range(3, target_resolution + 1):
        curr_g_block = block_G(res)
        curr_to_rgb_block = torgb(res)

        prev_images = x0
        for g in gen_block_list:
            prev_images = g(prev_images)

        curr_images = curr_g_block(prev_images)
        curr_images = curr_to_rgb_block(curr_images)

        prev_images = prev_to_rgb_block(prev_images)
        prev_images = layers.UpSampling2D(name="upsample_%dx%d" % (2**res, 2**res))(prev_images)

        images_out = FadeIn(alpha=alpha, name="fade_in_%dx%d" % (2**res, 2**res))([prev_images, curr_images])
        mdl = Model(inputs=x0, outputs=images_out)
        mdl.alpha = alpha
        model_list.append(mdl)
        gen_block_list.append(curr_g_block)

        prev_g_block = curr_g_block
        prev_to_rgb_block = curr_to_rgb_block

    # build final model
    x = x0
    for g in gen_block_list:
        x = g(x)
    x = curr_to_rgb_block(x)
    final_mdl = Model(inputs=x0, outputs=x)
    model_list.append(final_mdl)
    return model_list


def fromrgb(res, num_channels=3):
    x0 = layers.Input(shape=(2**res, 2**res, num_channels))
    #     x = layers.Conv2D(filters=nf(res - 1), kernel_size=1, padding="same")(x0)
    x = EqualizedLRConv2D(filters=nf(res - 1), kernel_size=1)(x0)
    x = ApplyBias()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return Model(inputs=x0, outputs=x, name="from_rgb_%dx%d" % (2**res, 2**res))


def block_D(res, mbstd_group_size=4):
    x0 = layers.Input(shape=(2**res, 2**res, nf(res - 1)))
    if res >= 3:
        x = x0
        for i in range(2):
            #             x = layers.Conv2D(filters=nf(res - (i + 1)), kernel_size=3, padding="same")(x)
            x = EqualizedLRConv2D(filters=nf(res - (i + 1)))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.AveragePooling2D()(x)
    else:
        if mbstd_group_size > 1:
            x = MiniBatchStd(mbstd_group_size)(x0)
            #             x = layers.Conv2D(filters=nf(res - 1), kernel_size=3, padding="same")(x)
            x = EqualizedLRConv2D(filters=nf(res - 1))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)

            x = layers.Flatten()(x)
            #             x = layers.Dense(units=nf(res - 2))(x)
            x = EqualizedLRDense(units=nf(res - 2))(x)
            x = ApplyBias()(x)
            x = layers.LeakyReLU(alpha=0.2)(x)

            #             x = layers.Dense(units=1)(x)
            x = EqualizedLRDense(units=1, gain=1.0)(x)
            x = ApplyBias()(x)
    return Model(inputs=x0, outputs=x, name="d_block_%dx%d" % (2**res, 2**res))


def build_D(target_resolution=10):
    model_list = list()
    disc_block_list = list()
    alpha = tf.Variable(initial_value=1.0, dtype='float32', trainable=False)
    for res in range(2, target_resolution + 1):
        x0 = layers.Input(shape=(2**res, 2**res, 3))
        curr_from_rgb = fromrgb(res)
        curr_D_block = block_D(res)

        x = curr_from_rgb(x0)
        x = curr_D_block(x)

        if res > 2:
            x_ds = layers.AveragePooling2D(name="downsample_%dx%d" % (2**res, 2**res))(x0)
            x_ds = prev_from_rgb(x_ds)
            x = FadeIn(alpha=alpha, name="fade_in_%dx%d" % (2**res, 2**res))([x_ds, x])
            for prev_d in disc_block_list[::-1]:
                x = prev_d(x)
            mdl = Model(inputs=x0, outputs=x)
            mdl.alpha = fade_in.alpha
            model_list.append(mdl)
        else:
            mdl = Model(inputs=x0, outputs=x)
            mdl.alpha = alpha
            model_list.append(mdl)

        disc_block_list.append(curr_D_block)
        prev_from_rgb = curr_from_rgb
    return model_list
