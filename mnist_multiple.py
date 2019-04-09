#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):

        in1 = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        in2 = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        conv1 = tf.keras.layers.Conv2D(filters=10,kernel_size=3,strides=2,padding='valid',activation=tf.nn.relu)
        conv1_a = conv1(in1)
        conv1_b = conv1(in2)
        conv2 = tf.keras.layers.Conv2D(filters=20,kernel_size=3,strides=2,padding='valid',activation=tf.nn.relu)
        conv2_a = conv2(conv1_a)
        conv2_b = conv2(conv1_b)
        flatten = tf.keras.layers.Flatten()
        flatten_a = flatten(conv2_a)
        flatten_b = flatten(conv2_b)
        dense1 = tf.keras.layers.Dense(200)
        dense1_a = dense1(flatten_a)
        dense1_b = dense1(flatten_b)
        out1 = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)
        predict_a = out1(dense1_a)
        predict_b = out1(dense1_b)
        conc = tf.keras.layers.Concatenate()([dense1_a, dense1_b])
        dense2 = tf.keras.layers.Dense(200, activation=tf.nn.relu)(conc)
        comparator = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense2)

        self.model = tf.keras.Model(inputs=[in1, in2], outputs=[predict_a, predict_b, comparator])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy"],
            metrics=["sparse_categorical_accuracy", "sparse_categorical_accuracy", "binary_accuracy"],
        )

    @staticmethod
    def _prepare_batches(batches_generator):
        batches = []
        for batch in batches_generator:
            batches.append(batch)
            if len(batches) >= 2:
                model_inputs = []
                model_targets = []
                for i in range(0, len(batches[0]["images"])):
                    image_a = batches[0]["images"][i]
                    image_b = batches[1]["images"][i]
                    label_a = batches[0]["labels"][i]
                    label_b = batches[1]["labels"][i]
                    comparison = 1 if label_a > label_b else 0
                    model_inputs.append([image_a, image_b])
                    model_targets.append([label_a, label_b, comparison])
                yield (model_inputs, model_targets)
                batches.clear()

    def train(self, mnist, args):
        for epoch in range(args.epochs):
            # TODO: Train for one epoch using `model.train_on_batch` for each batch.
            for batch in self._prepare_batches(mnist.train.batches(args.batch_size)):

                pass

            # Print development evaluation
            print("Dev {}: directly predicting: {:.4f}, comparing digits: {:.4f}".format(epoch + 1, *self.evaluate(mnist.dev, args)))

    def evaluate(self, dataset, args):
        # TODO: Evaluate the given dataset, returning two accuracies, the first being
        # the direct prediction of the model, and the second computed by comparing predicted
        # labels of the images.
        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            pass

        return direct_accuracy, indirect_accuracy


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)
    with open("mnist_multiple.out", "w") as out_file:
        direct, indirect = network.evaluate(mnist.test, args)
        print("{:.2f} {:.2f}".format(100 * direct, 100 * indirect), file=out_file)
