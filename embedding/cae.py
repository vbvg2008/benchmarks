import pdb
import random
import tempfile

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import image
from PIL import Image, ImageOps, ImageTransform
from tensorflow.keras import layers
from torch.utils.data import Dataset

import fastestimator as fe
from fastestimator.dataset.data import cifair10, mnist
from fastestimator.op.numpyop import Delete, NumpyOp
from fastestimator.op.numpyop.meta import OneOf
from fastestimator.op.numpyop.multivariate import GridDistortion, HorizontalFlip, RandomResizedCrop, Resize
from fastestimator.op.numpyop.univariate import CoarseDropout, ColorJitter, GaussianBlur, RandomRain, RandomShapes, \
    RandomSnow
from fastestimator.op.tensorop.loss import MeanSquaredError
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import ModelSaver


class RandomPair(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        paired_idx = random.randint(0, len(self.ds) - 1)
        img1 = self.ds[idx]["x"]
        img2 = self.ds[paired_idx]["x"]
        label1 = self.ds[idx]["y"]
        label2 = self.ds[paired_idx]["y"]
        return {"x1": img1, "x2": img2, "y1": label1, "y2": label2}


class Solarize(NumpyOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.loss_limit = 256

    def forward(self, data, state):
        threshold = 256 - round(random.uniform(0, self.loss_limit))
        return [self.transform_slice(x, threshold) for x in data]

    def transform_slice(self, data, threshold):
        data = np.where(data < threshold, data, 255 - data)
        return data


class Posterize(NumpyOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.bit_loss_limit = 7

    def forward(self, data, state):
        bits_to_keep = 8 - round(random.uniform(0, self.bit_loss_limit))
        return [self.transform_slice(x, bits_to_keep) for x in data]

    def transform_slice(self, data, bits_to_keep):
        image = Image.fromarray(data)
        image = ImageOps.posterize(image, bits_to_keep)
        return np.copy(np.asarray(image))


class RandomShape(RandomShapes):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         mode=mode,
                         transparency_range=(1.0, 1.0),
                         intensity_range=(122, 254))
        super().set_rua_level(1.0)

    def forward(self, data, state):
        shape_image = self.transform_slice(np.zeros_like(data[0]), state)
        alpha = np.random.uniform(0.1, 0.9)
        return [self.blend_slice(im, shape_image, alpha) for im in data]

    def blend_slice(self, data, shape, alpha):
        blend = shape * alpha + data * (1.0 - alpha)
        overlay = np.where(shape > 0, blend, data)
        return overlay.astype("uint8")

    def transform_slice(self, data, state):
        random_shaped_image = super().forward(data=[data], state=state)[0]
        return random_shaped_image


#TODO: This is dataset dependent
class GaussianBlurring(GaussianBlur):
    def __init__(self, inputs=None, outputs=None, mode=None):
        min_blur_limit, max_blur_limit = 3, 19
        super().__init__(inputs=inputs, outputs=outputs, mode=mode, blur_limit=(min_blur_limit, max_blur_limit))

    def forward(self, data, state):
        transformed_slice = self.replay_func(image=data[0])
        return [self.replay_func.replay(transformed_slice['replay'], image=im)['image'] for im in data]


class ColorJittering(ColorJitter):
    def __init__(self, inputs=None, outputs=None, mode=None):
        factor = 0.9
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         mode=mode,
                         brightness=factor,
                         contrast=factor,
                         saturation=factor,
                         hue=factor / 2)

    def forward(self, data, state):
        transformed_slice = self.replay_func(image=data[0])
        return [self.replay_func.replay(transformed_slice['replay'], image=im)['image'] for im in data]


# TODO: dataset dependent
class Cutout(CoarseDropout):
    def __init__(self, image_w, image_h, inputs=None, outputs=None, mode=None):
        min_width, min_height = image_w // 8, image_h // 8
        max_width, max_height = image_w // 4, image_h // 4
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         mode=mode,
                         max_holes=1,
                         max_height=max_height,
                         min_height=min_height,
                         max_width=max_width,
                         min_width=min_width)

    def forward(self, data, state):
        transformed_slice = self.replay_func(image=data[0])
        return [self.replay_func.replay(transformed_slice['replay'], image=im)['image'] for im in data]


class RandomEffectRain(RandomRain):
    def forward(self, data, state):
        transformed_slice = self.replay_func(image=data[0])
        return [self.replay_func.replay(transformed_slice['replay'], image=im)['image'] for im in data]


class RandomEffectSnow(RandomSnow):
    def forward(self, data, state):
        transformed_slice = self.replay_func(image=data[0])
        return [self.replay_func.replay(transformed_slice['replay'], image=im)['image'] for im in data]


class MyHorizontalFlip(HorizontalFlip):
    def __init__(self, inputs, outputs, mode=None):
        super().__init__(image_in=inputs[0], image_out=outputs[0], mask_in=inputs[1], mask_out=outputs[1], mode=mode)

    def forward(self, data, state):
        image, mask = data
        new_im, new_ma = super().forward(data=[image, mask], state=state)
        return new_im, new_ma


class Rotate(NumpyOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.degree = 90

    def forward(self, data, state):
        degree = random.uniform(-self.degree, self.degree)
        return [self.transform_slice(x, degree) for x in data]

    def transform_slice(self, data, degree):
        im = Image.fromarray(data)
        im = im.rotate(degree)
        return np.copy(np.asarray(im))


class ShearX(NumpyOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shear_coef = 0.3

    def forward(self, data, state):
        shear_coeff = random.uniform(-self.shear_coef, self.shear_coef)
        return [self.transform_slice(x, shear_coeff) for x in data]

    def transform_slice(self, data, shear_coeff):
        im = Image.fromarray(data)
        width, height = im.size
        xshift = round(abs(shear_coeff) * width)
        new_width = width + xshift
        im = im.transform((new_width, height),
                          ImageTransform.AffineTransform(
                              (1.0, shear_coeff, -xshift if shear_coeff > 0 else 0.0, 0.0, 1.0, 0.0)),
                          resample=Image.BICUBIC)
        im = im.resize((width, height))
        return np.copy(np.asarray(im))


class ShearY(NumpyOp):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.shear_coef = 0.3

    def forward(self, data, state):
        shear_coeff = random.uniform(-self.shear_coef, self.shear_coef)
        return [self.transform_slice(x, shear_coeff) for x in data]

    def transform_slice(self, data, shear_coeff):
        im = Image.fromarray(data)
        width, height = im.size
        yshift = round(abs(shear_coeff) * height)
        newheight = height + yshift
        im = im.transform((width, newheight),
                          ImageTransform.AffineTransform(
                              (1.0, 0.0, 0.0, shear_coeff, 1.0, -yshift if shear_coeff > 0 else 0.0)),
                          resample=Image.BICUBIC)
        im = im.resize((width, height))
        return np.copy(np.asarray(im))


class TranslateX(NumpyOp):
    def forward(self, data, state):
        displacement = random.uniform(-1, 1)
        return [self.transform_slice(x, displacement) for x in data]

    def transform_slice(self, data, displacement):
        im = Image.fromarray(data)
        width, height = im.size
        displacement = displacement * width / 3
        im = im.transform((width, height),
                          ImageTransform.AffineTransform((1.0, 0.0, displacement, 0.0, 1.0, 0.0)),
                          resample=Image.BICUBIC)
        return np.copy(np.asarray(im))


class TranslateY(NumpyOp):
    def forward(self, data, state):
        displacement = random.uniform(-1, 1)
        return [self.transform_slice(x, displacement) for x in data]

    def transform_slice(self, data, displacement):
        im = Image.fromarray(data)
        width, height = im.size
        displacement = displacement * height / 3
        im = im.transform((width, height),
                          ImageTransform.AffineTransform((1.0, 0.0, 0.0, 0.0, 1.0, displacement)),
                          resample=Image.BICUBIC)
        return np.copy(np.asarray(im))


class Scale(RandomResizedCrop):
    def __init__(self, image_h, image_w, inputs=None, outputs=None, mode=None):
        super().__init__(height=image_h,
                         width=image_w,
                         image_in=inputs[0],
                         image_out=outputs[0],
                         mask_in=inputs[1],
                         mask_out=outputs[1],
                         scale=(0.08, 1.0),
                         interpolation=cv2.INTER_CUBIC,
                         mode=mode)

    def forward(self, data, state):
        image, mask = data
        new_im, new_ma = super().forward(data=[image, mask], state=state)
        return new_im, new_ma


class MyGridDistort(GridDistortion):
    def __init__(self, inputs=None, outputs=None, mode=None):
        super().__init__(image_in=inputs[0], image_out=outputs[0], mask_in=inputs[1], mask_out=outputs[1], mode=mode)

    def forward(self, data, state):
        image, mask = data
        new_im, new_ma = super().forward(data=[image, mask], state=state)
        return new_im, new_ma


class RandomPair(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        paired_idx = random.randint(0, len(self.ds) - 1)
        img1 = self.ds[idx]["x"]
        img2 = self.ds[paired_idx]["x"]
        label1 = self.ds[idx]["y"]
        label2 = self.ds[paired_idx]["y"]
        return {"x1": img1, "x2": img2, "y1": label1, "y2": label2}


def encoder(input_size=(None, None, 3)):
    inputs = layers.Input(shape=input_size)
    # Stage1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    # Stage2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    # Stage3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    # Stage4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tfa.layers.AdaptiveAveragePooling2D(output_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def decoder(encoder, input_size=(None, None, 3), em_dim=512):
    img2_orig = layers.Input(shape=input_size)
    height = tf.shape(img2_orig)[1]
    width = tf.shape(img2_orig)[2]
    aug_embedding = layers.Input(shape=(em_dim, ))
    vec_diff_2 = layers.Dense(1024)(aug_embedding)
    vec2_orig = encoder(img2_orig)
    vec2_aug = vec2_orig + vec_diff_2
    vec2_spatial = layers.Dense(1024)(vec2_aug)
    x = layers.Reshape(target_shape=(2, 2, 256))(vec2_spatial)
    x = tf.image.resize(x, size=(width // 8, height // 8))  # resize to the original feature size
    # stage 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(interpolation='bilinear')(x)
    # stage 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(interpolation='bilinear')(x)
    # stage 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(interpolation='bilinear')(x)
    # stage 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    # to rgb
    x = layers.Conv2D(3, 1, activation='tanh')(x)
    return tf.keras.Model(inputs=[aug_embedding, img2_orig], outputs=x)


#TODO: maybe do not need to extra dense, directly make aug_embedding = vec_aug - vec_diff
def aug_embedding(em_dim=512):
    img1_orig = layers.Input(shape=(None, None, 3))
    img1_aug = layers.Input(shape=(None, None, 3))
    img2_orig = layers.Input(shape=(None, None, 3))
    model_enc = encoder()
    vec1_orig = model_enc(img1_orig)
    vec1_aug = model_enc(img1_aug)
    vec_diff_1 = vec1_aug - vec1_orig
    aug_embedding = layers.Dense(em_dim)(vec_diff_1)
    model_dec = decoder(model_enc, em_dim=em_dim)
    img2_aug = model_dec([aug_embedding, img2_orig])
    model_overall = tf.keras.Model(inputs=[img1_orig, img1_aug, img2_orig], outputs=img2_aug)
    return model_overall, model_enc, model_dec


class ToColor(NumpyOp):
    def forward(self, data, state):
        return [self.convert(x) for x in data]

    def convert(self, data):
        return np.stack([data, data, data], axis=-1)


class Rescale(NumpyOp):
    def forward(self, data, state):
        return [self.convert(x) for x in data]

    def convert(self, data):
        return np.float32(data / 255)


def get_estimator(save_dir=tempfile.mkdtemp(), epochs=10):
    train_data_mnist, eval_data_mnist = mnist.load_data()
    train_data_mnist, eval_data_mnist = RandomPair(train_data_mnist), RandomPair(eval_data_mnist)
    train_data_cifar, eval_data_cifar = cifair10.load_data()
    train_data_cifar, eval_data_cifar = RandomPair(train_data_cifar), RandomPair(eval_data_cifar)
    aug_options = [
        Solarize(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        Posterize(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        RandomShape(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        GaussianBlurring(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        ColorJittering(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        Cutout(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug"), image_w=32, image_h=32),
        RandomEffectRain(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        RandomEffectSnow(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        MyHorizontalFlip(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        Rotate(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        ShearX(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        ShearY(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        TranslateX(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        TranslateY(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug")),
        Scale(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug"), image_w=32, image_h=32),
        MyGridDistort(inputs=("x1", "x2"), outputs=("x1_aug", "x2_aug"))
    ]
    pipeline = fe.Pipeline(
        train_data={
            "mnist": train_data_mnist, "cifar": train_data_cifar
        },
        eval_data={
            "mnist": eval_data_mnist, "cifar": eval_data_cifar
        },
        batch_size=256,
        ops=[
            ToColor(inputs=("x1", "x2"), outputs=("x1", "x2"), ds_id="mnist"),
            Resize(height=32, width=32, image_in="x1", image_out="x1", ds_id="mnist"),
            Resize(height=32, width=32, image_in="x2", image_out="x2", ds_id="mnist"),
            OneOf(*aug_options),
            Rescale(inputs=("x1", "x1_aug", "x2", "x2_aug"), outputs=("x1", "x1_aug", "x2", "x2_aug")),
            Delete(keys=("y1", "y2"))
        ])
    model, enc, dec = fe.build(model_fn=aug_embedding, optimizer_fn=("adam", None, None))
    network = fe.Network(ops=[
        ModelOp(model=model, inputs=("x1", "x1_aug", "x2"), outputs="x2_aug_pred"),
        MeanSquaredError(inputs=("x2_aug_pred", "x2_aug"), outputs="mse_loss"),
        UpdateOp(model=model, loss_name="mse_loss")
    ])
    estimator = fe.Estimator(pipeline=pipeline,
                             network=network,
                             epochs=epochs,
                             train_steps_per_epoch=1000,
                             traces=ModelSaver(model=model, save_dir=save_dir, frequency=epochs))
    return estimator
