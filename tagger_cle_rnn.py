#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset


class Network:
    def __init__(self, args, num_words, num_tags, num_chars):
        word_ids = tf.keras.layers.Input(shape=[None])
        charseqs = tf.keras.layers.Input(shape=[None])
        charseq_ids = tf.keras.layers.Input(shape=[None], dtype=tf.int32)

        embed_charseqs = tf.keras.layers.Embedding(num_chars, args.cle_dim, mask_zero=True)(charseqs)
        gru_charseqs = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=args.cle_dim,
                                                                         return_sequences=False))(embed_charseqs)

        embed_cle = tf.keras.layers.Lambda(lambda args: tf.gather(args, charseq_ids))(gru_charseqs)
        embed_words = tf.keras.layers.Embedding(num_words, args.we_dim, mask_zero=True)(word_ids)

        concat = tf.keras.layers.Concatenate()([embed_words, embed_cle])

        if args.rnn_cell == "LSTM":
            bidir = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=args.rnn_cell_dim,
                                                                       return_sequences=True))(concat)
        else:
            bidir = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=args.rnn_cell_dim,
                                                                      return_sequences=True))(concat)

        predictions = tf.keras.layers.Dense(num_tags, activation=tf.nn.softmax)(bidir)

        self.model = tf.keras.Model(inputs=[word_ids, charseq_ids, charseqs], outputs=predictions)
        self._optimizer = tf.optimizers.Adam()
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._metrics = {"loss": tf.metrics.Mean(),
                         "accuracy": tf.metrics.SparseCategoricalAccuracy()}

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3,
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def train_batch(self, inputs, tags):
        mask = tf.cast(tf.not_equal(tags, tf.zeros(shape=tags.shape), dtype=tf.float32))
        with tf.GradientTape() as tape:
            probabilities = self.model(inputs, training=True)
            loss = self._loss(tags, probabilities, mask)
        gradients = tape.gradient(loss, self.model.variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.variables))

        tf.summary.experimental.set_step(self._optimizer.iterations)
        with self._writer.as_default():
            for name, metric in self._metrics.items():
                metric.reset_states()
                if name == "loss":
                    metric(loss)
                else:
                    metric.update_state(tags, probabilities, mask)
                tf.summary.scalar("train/{}".format(name), metric.result())

    def train_epoch(self, dataset, args):
        for batch in dataset.batches(args.batch_size):
            self.train_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids,
                              batch[dataset.FORMS].charseqs], batch[dataset.TAGS].word_ids)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3,
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def evaluate_batch(self, inputs, tags):
        mask = tf.cast(tf.not_equal(tags, tf.zeros(shape=tags.shape), dtype=tf.float32))
        probabilities = self.model(inputs, training=False)
        loss = self._loss(tags, probabilities, mask)
        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            else:
                metric.update_state(tags, probabilities, mask)

    def evaluate(self, dataset, dataset_name, args):
        for metric in self._metrics.values():
            metric.reset_states()
        for batch in dataset.batches(args.batch_size):
            self.evaluate_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids,
                                 batch[dataset.FORMS].charseqs], batch[dataset.TAGS].word_ids)
        metrics = {name: metric.result() for name, metric in self._metrics.items()}
        with self._writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar("{}/{}".format(dataset_name, name), value)

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
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--max_sentences", default=5000, type=int, help="Maximum number of sentences to load.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.initializers.glorot_uniform(seed=42)
        tf.keras.utils.get_custom_objects()["orthogonal"] = lambda: tf.initializers.orthogonal(seed=42)
        tf.keras.utils.get_custom_objects()["uniform"] = lambda: tf.initializers.RandomUniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the network and train
    network = Network(args,
                      num_words=len(morpho.train.data[morpho.train.FORMS].words),
                      num_tags=len(morpho.train.data[morpho.train.TAGS].words),
                      num_chars=len(morpho.train.data[morpho.train.FORMS].alphabet))
    #network.plot()

    for epoch in range(args.epochs):
        network.train_epoch(morpho.train, args)
        metrics = network.evaluate(morpho.dev, "dev", args)

    metrics = network.evaluate(morpho.test, "test", args)
    with open("tagger_we.out", "w") as out_file:
        print("{:.2f}".format(100 * metrics["accuracy"]), file=out_file)

# 698f4a25-47cc-11e9-b0fd-00505601122b
# b5770ea9-40bc-11e9-b0fd-00505601122b

