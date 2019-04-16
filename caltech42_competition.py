#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub

from caltech42 import Caltech42

# The neural network model
class Network:
    def __init__(self, args, download):
        inputs = tf.keras.layers.Input(shape=[Caltech42.MIN_SIZE, Caltech42.MIN_SIZE, Caltech42.C])
        if download:
            mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2",
                                         output_shape=[1280], trainable=False)
            features = mobilenet(inputs, training=False)
            # dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(features)
            output = tf.keras.layers.Dense(Caltech42.LABELS, activation=tf.nn.softmax)(features)
            self.model = tf.keras.Model(inputs=inputs, outputs=output)
        else:
            self.model = tf.keras.experimental.load_from_saved_model("finetuned", {"KerasLayer": tfhub.KerasLayer})

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, caltech42, args):
        self.model.fit(
            caltech42.train.data["images"], caltech42.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(caltech42.dev.data["images"], caltech42.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    @staticmethod
    def image_process(image):
        im = tf.image.decode_image(image, channels=3, dtype=tf.float32)
        im = tf.image.resize(im, size=(224, 224))
        return im.numpy()

    def predict(self, caltech42, args):
        predicts = self.model.predict(caltech42.test.data["images"], batch_size=args.batch_size)
        return predicts

    def plot(self):
        tf.keras.utils.plot_model(self.model, to_file=__file__[:-3] + '_model.png', show_shapes=True)

    def save(self):
        path = "finetuned"
        tf.keras.experimental.export_saved_model(self.model, path, serving_only=True)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
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

    # Load data
    caltech42 = Caltech42(Network.image_process)

    # Create the network and train
    network = Network(args, True)
    # network.plot()
    network.train(caltech42, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(caltech42, args):
            print(np.argmax(probs), file=out_file)

# 698f4a25-47cc-11e9-b0fd-00505601122b
# b5770ea9-40bc-11e9-b0fd-00505601122b
