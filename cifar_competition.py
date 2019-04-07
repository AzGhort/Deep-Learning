#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10


# The neural network model
class WideNet(tf.keras.Model):

    def __init__(self, args):
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
        # 32x32x16
        conv1 = tf.keras.layers.Conv2D(filters=16,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)(inputs)
        # 32x32x32
        conv2 = self.add_convolution_group(2, 4, 16, [3, 3], 'same', conv1)
        # 16x16x64
        # reshape residual info
        conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=2, padding='same', use_bias=False)(conv2)
        conv3 = self.add_convolution_group(2, 4, 32, [3, 3], 'same', conv2)
        # 8x8x128
        # reshape residual info
        conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=2, padding='same', use_bias=False)(conv3)
        conv4 = self.add_convolution_group(2, 4, 64, [3, 3], 'same', conv3)

        avg_pool = tf.keras.layers.AveragePooling2D(pool_size=8)(conv4)
        outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(avg_pool)

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def add_convolution_group(self, k, depth, channels, kernels, padd, previous):
        layer = previous
        for _ in range(depth):
            layer = self.add_residual_block(kernels, channels, k, padd, layer)
        return layer

    def add_bn_relu_conv(self, filt, kernel, padd, previous):
        first = tf.keras.layers.BatchNormalization(axis=3)(previous)
        second = tf.keras.layers.Activation(activation=tf.nn.relu)(first)
        layer = tf.keras.layers.Conv2D(filters=filt, kernel_size=kernel, strides=1,
                                       padding=padd, use_bias=False, activation=None)(second)
        return layer

    def add_residual_block(self, kernels, filt, k, padd, previous):
        layers = [self.add_bn_relu_conv(filt, kernels[0], padd, previous) for _ in range(k)]
        for kernel in kernels[1:]:
            layers = [self.add_bn_relu_conv(filt, kernel, padd, layer) for layer in layers]
        layers.append(previous)
        hidden = tf.keras.layers.Add()(layers)
        return hidden

    def train(self, cifar, args):
        self.fit(
            cifar.train.data["images"], cifar.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )


# https://arxiv.org/pdf/1512.03385.pdf
class ResNet(tf.keras.Model):

    def __init__(self, layers):
        group_block_count = int((layers - 2)/6)
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
        # 32x32x16
        conv1 = tf.keras.layers.Conv2D(filters=16,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)(inputs)
        # 32x32x32
        conv2 = self.add_convolution_group(group_block_count, 16, False, [3, 3], 'same', conv1)
        # 16x16x64
        conv3 = self.add_convolution_group(group_block_count, 32, True, [3, 3], 'same', conv2)
        # 8x8x128
        conv4 = self.add_convolution_group(group_block_count, 64, True, [3, 3], 'same', conv3)

        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(conv4)
        outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(avg_pool)

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def add_convolution_group(self, blocks_count, filters, downsample, res_block_kernels, padd, previous):
        layer = previous
        if downsample: # downsample the image
            layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding='same', use_bias=False)(layer)
        for _ in range(blocks_count):
            layer = self.add_residual_block(res_block_kernels, filters, padd, layer)
        return layer

    def add_conv_bn_relu(self, filt, kernel, padd, previous):
        first = tf.keras.layers.Conv2D(filters=filt, kernel_size=kernel, strides=1,
                                       padding=padd, use_bias=False, activation=None)(previous)
        second = tf.keras.layers.BatchNormalization(axis=3)(first)
        layer = tf.keras.layers.Activation(activation=tf.nn.relu,)(second)
        return layer

    def add_residual_block(self, kernels, filt, padd, previous):
        layer = previous
        for kernel in kernels:
            layer = self.add_conv_bn_relu(filt, kernel, padd, layer)
        hidden = tf.keras.layers.Add()([layer, previous])
        return hidden

    def train(self, tr_i, tr_o, dev_i, dev_o, args):
        self.fit(
            tr_i, tr_o,
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(dev_i, dev_o),
            callbacks=[self.tb_callback],
        )


class DataAugment:

    @staticmethod
    def cutout_image_dataset(dataset, image_size, cutout_size, versions):
        images = dataset["images"]
        labels = dataset["labels"]
        new_imgs = []
        new_lbls = []
        mean = np.mean(images)
        for image in images:
            counter = 0
            for _ in range(versions):
                x = int(np.random.rand()*image_size)
                y = int(np.random.rand()*image_size)
                end_x = x + cutout_size if (x + cutout_size <= image_size) else image_size
                end_y = y + cutout_size if (y + cutout_size <= image_size) else image_size
                cutout = np.ones((end_x - x, end_y - y, 3))*mean
                new_im = np.copy(image)
                new_label = np.copy(labels[counter])
                new_im[x:end_x, y:end_y] = cutout
                new_imgs.append(new_im)
                new_lbls.append(new_label)
            counter += 1
        new_imgs = np.array(new_imgs)
        new_lbls = np.array(new_lbls)
        input = np.append(images, new_imgs, axis=0)
        output = np.append(labels, new_lbls, axis=0)
        return input, output


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--layers", default=54, type=int, help="Number of layers.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}-{}".format(
        os.path.basename(__file__),
        "ResNet" + str(args.layers),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    # train_in, train_out = DataAugment.cutout_image_dataset(cifar.train.data, 32, 16, 3)
    # test_in, test_out = DataAugment.cutout_image_dataset(cifar.test.data, 32, 16, 3)
    # dev_in, dev_out = DataAugment.cutout_image_dataset(cifar.dev.data, 32, 16, 3)

    # Create the network and train
    network = ResNet(args.layers)
    network.train(cifar.train.data["images"], cifar.train.data["labels"],
                  cifar.dev.data["images"], cifar.dev.data["labels"], args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)

# 698f4a25-47cc-11e9-b0fd-00505601122b
# b5770ea9-40bc-11e9-b0fd-00505601122b
