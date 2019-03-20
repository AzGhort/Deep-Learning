#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    data_mapping = {}
    count = 0
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            count += 1
            if not (line in data_mapping):
                data_mapping[line] = 1
            else:
                data_mapping[line] += 1

    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    model_mapping = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            parts = line.split("\t")
            model_mapping[parts[0]] = parts[1]

    data_distribution = np.fromiter(data_mapping.values(), dtype=float)/count
    model_distribution = np.fromiter((model_mapping.get(key, 0) for key in data_mapping.keys()), dtype=float)

    # entropy
    entropy = -np.sum(data_distribution*np.log(data_distribution))
    print("{:.2f}".format(entropy))

    # cross-entropy
    cross_entropy = -np.sum(data_distribution*np.log(model_distribution))
    print("{:.2f}".format(cross_entropy))

    # KL divergence
    kl_divergence = np.sum(data_distribution*(np.log(data_distribution) - np.log(model_distribution)))
    print("{:.2f}".format(kl_divergence))
