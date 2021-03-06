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
        conv2 = tf.keras.layers.Conv2D(filters=20,kernel_size=3,strides=2,padding='valid',activation=tf.nn.relu)
        flatten = tf.keras.layers.Flatten()
        dense1 = tf.keras.layers.Dense(200, activation=tf.nn.relu)
        out1 = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

        # branch A
        conv1_a = conv1(in1)
        conv2_a = conv2(conv1_a)
        flatten_a = flatten(conv2_a)
        dense1_a = dense1(flatten_a)
        predict_a = out1(dense1_a)

        # branch B
        conv1_b = conv1(in2)
        conv2_b = conv2(conv1_b)
        flatten_b = flatten(conv2_b)
        dense1_b = dense1(flatten_b)
        predict_b = out1(dense1_b)

        # concatenated part
        conc = tf.keras.layers.Concatenate()([dense1_a, dense1_b])
        dense2 = tf.keras.layers.Dense(200, activation=tf.nn.relu)(conc)
        comparator = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense2)

        self.model = tf.keras.Model(inputs=[in1, in2], outputs=[predict_a, predict_b, comparator])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy"],
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="num_accuracy_a"),
                     tf.keras.metrics.SparseCategoricalAccuracy(name="num_accuracy_b"),
                     tf.keras.metrics.BinaryAccuracy(name="comp_accuracy")],
        )

    @staticmethod
    def _prepare_batches(batches_generator):
        batches = []
        for batch in batches_generator:
            batches.append(batch)
            if len(batches) >= 2:
                model_inputs = [batches[0]["images"],  batches[1]["images"]]
                model_targets = [batches[0]["labels"],  batches[1]["labels"],
                                 np.asarray([1 if label_a > label_b else 0 for label_a, label_b
                                           in zip(batches[0]["labels"], batches[1]["labels"])])]
                yield (model_inputs, model_targets)
                batches.clear()

    def train(self, mnist, args):
        for epoch in range(args.epochs):
            for batch in self._prepare_batches(mnist.train.batches(args.batch_size)):
                self.model.train_on_batch(batch[0], batch[1])
            print("Dev {}: directly predicting: {:.4f}, comparing digits: {:.4f}".format(epoch + 1, *self.evaluate(mnist.dev, args)))

    def evaluate(self, dataset, args):
        direct_accuracy = 0
        indirect_accuracy = 0
        count = 0
        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            outputs = self.model.predict(inputs, batch_size=args.batch_size)
            direct_accuracy += np.sum([1 if (output > 0.5 and target == 1) or (output <= 0.5 and target == 0)
                                       else 0 for output, target in zip(outputs[2], targets[2])])
            output_labels_a = [np.argmax(a) for a in outputs[0]]
            output_labels_b = [np.argmax(b) for b in outputs[1]]
            indirect_accuracy += np.sum([1 if (target_a <= target_b and output_a <= output_b) or
                                              (target_a > target_b and output_a > output_b)
                                         else 0 for target_a, target_b, output_a, output_b in
                                         zip(targets[0], targets[1], output_labels_a, output_labels_b)])
            count += args.batch_size
        return direct_accuracy / count, indirect_accuracy / count

    def plot(self):
        tf.keras.utils.plot_model(self.model, to_file='mnist_multiple_model.png', show_shapes=True)


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
    #network.plot()
    with open("mnist_multiple.out", "w") as out_file:
        direct, indirect = network.evaluate(mnist.test, args)
        print("{:.2f} {:.2f}".format(100 * direct, 100 * indirect), file=out_file)

# 698f4a25-47cc-11e9-b0fd-00505601122b
# b5770ea9-40bc-11e9-b0fd-00505601122b
