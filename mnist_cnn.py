#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from mnist import MNIST


# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args):
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        start_r = args.cnn.find('[') + 1
        end_r = args.cnn.find(']', start_r)
        config = args.cnn[0:start_r-1] + args.cnn[end_r+1:] if (not(start_r == 0 and end_r == -1)) else args.cnn
        config_r = args.cnn[start_r:end_r]
        layers = config.split(",")
        hidden = inputs
        # add hidden layers
        for arg in layers:
            if arg[0] == "R":
                res = config_r.split(",")
                last_r = hidden
                for parallelLayer in res:
                    last_r = self.set_next_layer_section(last_r, parallelLayer)
                hidden = tf.keras.layers.Add()([hidden, last_r])
            else:
                hidden = self.set_next_layer_section(hidden, arg)
        # add last layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, mnist, args):
        self.fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def test(self, mnist, args):
        test_logs = self.evaluate(mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size)
        self.tb_callback.on_epoch_end(1, dict(("val_test_" + metric, value) for metric, value in zip(self.metrics_names, test_logs)))
        return test_logs[self.metrics_names.index("accuracy")]

    def set_next_layer_section(self, previous, arg):
        config = arg.split("-")
        type = config[0]
        layer = None
        if type == "C":
            layer = tf.keras.layers.Conv2D(filters=int(config[1]), kernel_size=int(config[2]), strides=int(config[3]),
                                           padding=config[4], activation=tf.nn.relu)(previous)
        elif type == "CB":
            first = tf.keras.layers.Conv2D(filters=int(config[1]), kernel_size=int(config[2]), strides=int(config[3]),
                                           padding=config[4], use_bias=False, activation=None)(previous)
            second = tf.keras.layers.BatchNormalization(axis=3)(first)
            layer = tf.keras.layers.Activation(activation=tf.nn.relu)(second)
        elif type == "M":
            layer = tf.keras.layers.MaxPool2D(pool_size=int(config[1]), strides=int(config[2]))(previous)
        elif type == "F":
            layer = tf.keras.layers.Flatten()(previous)
        elif type == "D":
            layer = tf.keras.layers.Dense(int(config[1]), activation=tf.nn.relu)(previous)
        return layer


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser() #"C-8-3-5-same,C-8-3-2-valid,F,D-50"
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.") #"C-8-3-5-valid,R-[C-8-3-1-same,C-8-3-1-same],F,D-50"
    parser.add_argument("--cnn", default="C-8-3-5-valid,R-[C-8-3-1-same,C-8-3-1-same],F,D-50", type=str, help="CNN architecture.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
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

    # Compute test set accuracy and print it
    accuracy = network.test(mnist, args)
    with open("mnist_cnn.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)

# 698f4a25-47cc-11e9-b0fd-00505601122b
# b5770ea9-40bc-11e9-b0fd-00505601122b
