#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from modelnet import ModelNet


# The neural network model
# based on VoxNet, https://arxiv.org/abs/1604.03351
class Network:
    def __init__(self, modelnet, args):
        inputs = tf.keras.layers.Input(shape=[ModelNet.D, ModelNet.H, ModelNet.W, ModelNet.C])

        # convolution group 1
        conv1 = tf.keras.layers.Conv3D(filters=32, kernel_size=5, strides=1,
                                       padding='valid', use_bias=False, activation=None)(inputs)
        bn1 = tf.keras.layers.BatchNormalization(axis=4)(conv1)
        act1 = tf.keras.layers.Activation(activation=tf.nn.relu)(bn1)

        # convolution group 2
        conv2 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1,
                                       padding='valid', use_bias=False, activation=None)(act1)
        bn2 = tf.keras.layers.BatchNormalization(axis=4)(conv2)
        act2 = tf.keras.layers.Activation(activation=tf.nn.relu)(bn2)

        # max pool and dense
        max_pool = tf.keras.layers.GlobalMaxPooling3D()(act2)
        #flat = tf.keras.layers.Flatten()(max_pool)
        dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(max_pool)
        outputs = tf.keras.layers.Dense(len(ModelNet.LABELS), activation=tf.nn.softmax)(dense1)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, modelnet, args):
        self.model.fit(
            modelnet.train.data["voxels"], modelnet.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(modelnet.dev.data["voxels"], modelnet.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def predict(self, dataset, args):
        probs = []
        for batch in dataset.batches(args.batch_size):
            out = self.model.predict_on_batch(batch["voxels"])
            for prob in out:
                probs.append(prob)
        return probs

    def plot(self):
        tf.keras.utils.plot_model(self.model, to_file=__file__[:-3] + '_model.png', show_shapes=True)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    modelnet = ModelNet(args.modelnet)

    # Create the network and train
    network = Network(modelnet, args)
    network.plot()
    network.train(modelnet, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "3d_recognition_test.txt"
    if os.path.isdir(args.logdir):
        out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for probs in network.predict(modelnet.test, args):
            print(np.argmax(probs), file=out_file)

# 698f4a25-47cc-11e9-b0fd-00505601122b
# b5770ea9-40bc-11e9-b0fd-00505601122b
