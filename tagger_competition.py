#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset


class Network:
    def __init__(self, pdt, args):
        word_ids = tf.keras.layers.Input(shape=[None])
        charseqs = tf.keras.layers.Input(shape=[None])
        charseq_ids = tf.keras.layers.Input(shape=[None], dtype=tf.int32)

        embed_charseqs = tf.keras.layers.Embedding(len(pdt.train.data[pdt.train.FORMS].alphabet),
                                                   32, mask_zero=True)(charseqs)
        gru_charseqs = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=32,
                                                                          return_sequences=False))(embed_charseqs)

        embed_cle = tf.keras.layers.Lambda(lambda args: tf.gather(args[0], args[1]))([gru_charseqs, charseq_ids])
        embed_words = tf.keras.layers.Embedding(len(pdt.train.data[pdt.train.FORMS].words),
                                                64, mask_zero=True)(word_ids)

        concat = tf.keras.layers.Concatenate()([embed_words, embed_cle])

        bidir = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=64, return_sequences=True))(concat)

        predictions = tf.keras.layers.Dense(len(pdt.train.data[pdt.train.TAGS].words), activation=tf.nn.softmax)(bidir)

        self.model = tf.keras.Model(inputs=[word_ids, charseq_ids, charseqs], outputs=predictions)
        self._optimizer = tf.optimizers.Adam()
        self._loss = tf.losses.SparseCategoricalCrossentropy()
        self._metrics = {"loss": tf.metrics.Mean(),
                         "accuracy": tf.metrics.SparseCategoricalAccuracy()}

        self.model.compile(optimizer=tf.optimizers.Adam(),
                           loss=tf.losses.SparseCategoricalCrossentropy(),
                           metrics={"loss": tf.metrics.Mean(), "accuracy": tf.metrics.SparseCategoricalAccuracy()}
                           #[tf.metrics.SparseCategoricalAccuracy(name="accuracy")]
                           )

        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3,
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def train_batch(self, inputs, tags):
        mask = tf.cast(tf.not_equal(tags, 0), dtype=tf.int32)
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
                tf.summary.scalar("train/{}".format(name), metric.result(), self._optimizer.iterations)

    def train_epoch(self, dataset, args):
        counter = 0
        for batch in dataset.batches(args.batch_size):
            #print("Training batch number " + str(counter))
            self.train_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids,
                              batch[dataset.FORMS].charseqs], batch[dataset.TAGS].word_ids)
            counter += 1

    @tf.function(input_signature=[[tf.TensorSpec(shape=[None, None], dtype=tf.int32)] * 3,
                                  tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    def evaluate_batch(self, inputs, tags):
        mask = tf.cast(tf.not_equal(tags, 0), dtype=tf.int32)
        probabilities = self.model(inputs, training=False)
        loss = self._loss(tags, probabilities, mask)
        for name, metric in self._metrics.items():
            if name == "loss":
                metric(loss)
            else:
                metric.update_state(tags, probabilities, mask)

    def evaluate(self, dataset, dataset_name, args):
        return self.model.evaluate([dataset.data[dataset.FORMS].word_ids, dataset.data[dataset.FORMS].charseq_ids,
                                    dataset.data[dataset.FORMS].charseqs], dataset.data[dataset.TAGS].word_ids)

    def predict(self, dataset, args):
        predicted = []
        for batch in dataset.batches(args.batch_size):
            out = self.model.predict_on_batch([batch[dataset.FORMS].word_ids, batch[dataset.FORMS].charseq_ids,
                                               batch[dataset.FORMS].charseqs])
            for pred_sentence in out:
                indices = []
                for pred_word in pred_sentence:
                    indices.append(np.argmax(pred_word))
                predicted.append(indices)
        return predicted

    def plot(self):
        tf.keras.utils.plot_model(self.model, to_file=__file__[:-3] + '_model.png', show_shapes=True)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=70, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=15, type=int, help="Maximum number of threads to use.")
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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Create the network and train
    network = Network(morpho, args)
    # network.plot()

    for epoch in range(args.epochs):
        print("Epoch " + str(epoch) + " started.")
        network.train_epoch(morpho.train, args)
        #metrics = network.evaluate(morpho.dev, "dev", args)
        print("Epoch " + str(epoch) + " ended.")
        #print("Dev set accuracy: " + metrics["loss"])
        #print("Dev set loss: " + metrics["accuracy"])

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    out_path = "tagger_competition_test.txt"
    if os.path.isdir(args.logdir):
        out_path = os.path.join(args.logdir, out_path)
    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, sentence in enumerate(network.predict(morpho.test, args)):
            for j in range(len(morpho.test.data[morpho.test.FORMS].word_strings[i])):
                print(morpho.test.data[morpho.test.FORMS].word_strings[i][j],
                      morpho.test.data[morpho.test.LEMMAS].word_strings[i][j],
                      morpho.test.data[morpho.test.TAGS].words[sentence[j]],
                      sep="\t", file=out_file)
            print(file=out_file)

# 698f4a25-47cc-11e9-b0fd-00505601122b
# b5770ea9-40bc-11e9-b0fd-00505601122b