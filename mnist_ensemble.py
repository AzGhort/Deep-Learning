#!/usr/bin/env python3
#!/usr/bin/env python3
import argparse

import numpy as np
import tensorflow as tf

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="200", type=str, help="Hidden layer configuration.")
parser.add_argument("--models", default=3, type=int, help="Number of models.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Load data
mnist = MNIST()

# Create models
models = []
for model in range(args.models):
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42 + model)

    models.append(tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
    ] + [tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu) for hidden_layer in args.hidden_layers] + [
        tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
    ]))

    models[-1].compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="individual_accuracy")],
    )

    print("Training model {}: ".format(model + 1), end="", flush=True)
    models[-1].fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs, verbose=0
    )
    print("Done")

with open("mnist_ensemble.out", "w") as out_file:
    for model in range(args.models):
        ind_logs = models[model].evaluate(mnist.dev.data["images"], mnist.dev.data["labels"], batch_size=args.batch_size)
        individual_accuracy = ind_logs[1]

        if model == 0:
            print("{:.2f} {:.2f}".format(100 * individual_accuracy, 100 * individual_accuracy), file=out_file)
            continue

        inputLayer = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        outs = [m(inputLayer) for m in models[0:model+1]]
        averagingLayer = tf.keras.layers.Average()(outs)
        averagedModel = tf.keras.Model(inputLayer, averagingLayer)

        averagedModel.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="ensembled_accuracy")],
        )

        ensemble_logs = averagedModel.evaluate(mnist.dev.data["images"],mnist.dev.data["labels"], batch_size=args.batch_size)
        ensemble_accuracy = ensemble_logs[1]

        # Print the results.
        print("{:.2f} {:.2f}".format(100 * individual_accuracy, 100 * ensemble_accuracy), file=out_file)
