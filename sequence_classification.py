#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# Dataset for generating sequences, with labels predicting whether the cumulative sum
# is odd/even.
class Dataset:
    def __init__(self, sequences_num, sequence_length, sequence_dim, seed, shuffle_batches=True):
        sequences = np.zeros([sequences_num, sequence_length, sequence_dim], np.int32)
        labels = np.zeros([sequences_num, sequence_length, 1], np.bool)
        generator = np.random.RandomState(seed)
        for i in range(sequences_num):
            sequences[i, :, 0] = generator.random_integers(0, max(1, sequence_dim - 1), size=[sequence_length])
            labels[i, :, 0] = np.bitwise_and(np.cumsum(sequences[i, :, 0]), 1)
            if sequence_dim > 1:
                sequences[i] = np.eye(sequence_dim)[sequences[i, :, 0]]
        self._data = {"sequences": sequences.astype(np.float32), "labels": labels}
        self._size = sequences_num

        self._shuffler = np.random.RandomState(seed) if shuffle_batches else None

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._size

    def batches(self, size=None):
        permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
        while len(permutation):
            batch_size = min(size or np.inf, len(permutation))
            batch_perm = permutation[:batch_size]
            permutation = permutation[batch_size:]

            batch = {}
            for key in self._data:
                batch[key] = self._data[key][batch_perm]
            yield batch


class Network:
    def __init__(self, args):
        sequences = tf.keras.layers.Input(shape=[args.sequence_length, args.sequence_dim])

        hidden = self.set_rnn_layer_from_args(args, sequences)

        if args.hidden_layer is not None:
            hidden = tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu)(hidden)

        predictions = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)

        self.model = tf.keras.Model(inputs=sequences, outputs=predictions)

        self._optimizer = tf.keras.optimizers.Adam()
        self._loss = tf.keras.losses.BinaryCrossentropy()
        self._metrics = {"loss": tf.keras.metrics.Mean(),
                         "accuracy": tf.keras.metrics.BinaryAccuracy()}

        self._writer = tf.summary.create_file_writer(
            args.logdir,
            flush_millis=10000
        )

    @staticmethod
    def set_rnn_layer_from_args(args, previous):
        if args.rnn_cell == "LSTM":
            return tf.keras.layers.LSTM(units=args.rnn_cell_dim, return_sequences=True)(previous)
        elif args.rnn_cell == "SimpleRNN":
            return tf.keras.layers.SimpleRNN(units=args.rnn_cell_dim, return_sequences=True)(previous)
        elif args.rnn_cell == "GRU":
            return tf.keras.layers.GRU(units=args.rnn_cell_dim, return_sequences=True)(previous)
        else:
            return None

    # @tf.function
    def train_on_batch(self, batch, clip_gradient):
        with tf.GradientTape() as tape:
            probabilities = self.model(batch["sequences"], training=True)
            loss = self._loss(batch["labels"], probabilities)
        gradients = tape.gradient(loss, self.model.trainable_variables)

        gradient_norm = tf.linalg.global_norm(gradients)
        if clip_gradient is not None:
            (gradients, gradient_norm) = tf.clip_by_global_norm(gradients, clip_norm=clip_gradient)

        grads_and_vars = zip(gradients, self._optimizer.variables())
        self._optimizer.apply_gradients(grads_and_vars)

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss":
                    metric.update_state(loss)
                else:
                    metric.update_state(batch["labels"], probabilities)
                tf.summary.scalar("train/" + name, metric.result(), self._optimizer.iterations)
            tf.summary.scalar("train/gradient_norm", gradient_norm, self._optimizer.iterations)

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            self.train_on_batch(batch, args.clip_gradient)

    @tf.function
    def predict_batch(self, batch):
        return self.model(batch["sequences"], training=False)

    def evaluate(self, dataset, args):
        for metric in self._metrics:
            metric.reset_states()
        for batch in dataset.batches(args.batch_size):
            predictions = self.predict_batch(batch)
            loss = self._loss(batch["labels"], predictions)
            for name, metric in self._metrics.items():
                if name == "loss":
                    metric.update_state(loss)
                else:
                    metric.update_state(batch["labels"], predictions)

        metrics = {name: value for name, value in self._metrics.items()}
        with self._writer.as_default():
            for name, metric in metrics.items():
                tf.summary.scalar("test/" + name, metric)
        return metrics

    def plot(self):
        tf.keras.utils.plot_model(self.model, to_file=__file__[:-3]+ '_model.png', show_shapes=True)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--clip_gradient", default=None, type=lambda x: None if x == "None" else float(x), help="Gradient clipping norm."),
    parser.add_argument("--hidden_layer", default=None, type=lambda x: None if x == "None" else int(x), help="Dense layer after RNN.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=10, type=int, help="RNN cell dimension.")
    parser.add_argument("--sequence_dim", default=1, type=int, help="Sequence element dimension.")
    parser.add_argument("--sequence_length", default=50, type=int, help="Sequence length.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--test_sequences", default=1000, type=int, help="Number of testing sequences.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--train_sequences", default=10000, type=int, help="Number of training sequences.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
        tf.keras.utils.get_custom_objects()["orthogonal"] = lambda: tf.keras.initializers.orthogonal(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Create the data
    train = Dataset(args.train_sequences, args.sequence_length, args.sequence_dim, seed=42, shuffle_batches=True)
    test = Dataset(args.test_sequences, args.sequence_length, args.sequence_dim, seed=43, shuffle_batches=False)

    # Create the network and train
    network = Network(args)
    #network.plot()
    for epoch in range(args.epochs):
        network.train_epoch(train, args)
        metrics = network.evaluate(test, args)
    with open("sequence_classification.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["accuracy"]), file=out_file)

# 698f4a25-47cc-11e9-b0fd-00505601122b
# b5770ea9-40bc-11e9-b0fd-00505601122b
