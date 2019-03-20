#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST


def categorical_to_one_hot_smoothed(distribution, alfa):
    smoothed = tf.keras.utils.to_categorical(distribution)
    smoothed += (np.ones(shape=smoothed.shape) * alfa)/smoothed.shape[0]
    smoothed[distribution] -= (np.ones(shape=smoothed.shape) * alfa)/smoothed.shape[0] + alfa
    return smoothed


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0, type=float, help="Dropout regularization.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
parser.add_argument("--l2", default=0, type=float, help="L2 regularization.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
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

# Load data
mnist = MNIST()

regularizer = None
if args.l2 != 0:
    regularizer = tf.keras.regularizers.L1L2(l2=args.l2)

# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]))
if args.dropout > 0:
    model.add(tf.keras.layers.Dropout(rate=args.dropout))
for hidden_layer in args.hidden_layers:
    model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu),
              kernel_regularizer=regularizer, bias_regularizer=regularizer)
    if args.dropout > 0:
        model.add(tf.keras.layers.Dropout(rate=args.dropout))
model.add(tf.keras.layers.Dense(MNIST.LABELS, kernel_regularizer=regularizer, bias_regularizer=regularizer))

loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics_ = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
train_i = mnist.train.data["images"]
train_o = mnist.train.data["labels"]
dev_i = mnist.dev.data["images"]
dev_o = mnist.dev.data["labels"]
test_i = mnist.test.data["images"]
test_o = mnist.test.data["labels"]

if args.label_smoothing > 0:
    loss_ = tf.keras.losses.CategoricalCrossentropy()
    metrics_ = tf.keras.metrics.CategoricalAccuracy()
    train_o = [categorical_to_one_hot_smoothed(o, args.label_smoothing) for o in train_o]
    dev_o = [categorical_to_one_hot_smoothed(o, args.label_smoothing) for o in dev_o]
    test_o = [categorical_to_one_hot_smoothed(o, args.label_smoothing) for o in test_o]

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=loss_,
    metrics=metrics_,
)

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None
model.fit(
    train_i[:5000], train_o[:5000],
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(dev_i, dev_o),
    callbacks=[tb_callback],
)

test_logs = model.evaluate(test_i, test_o, batch_size=args.batch_size)
tb_callback.on_epoch_end(1, dict(("val_test_" + metric, value) for metric, value in zip(model.metrics_names, test_logs)))

accuracy = test_logs[model.metrics_names.index("accuracy")]
with open("mnist_regularization.out", "w") as out_file:
    print("{:.2f}".format(100 * accuracy), file=out_file)