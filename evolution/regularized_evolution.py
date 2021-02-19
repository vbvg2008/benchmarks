import random

import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.dataset.data import cifar10
from fastestimator.op.numpyop.univariate import ExpandDims, Minmax
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.metric import Accuracy
from tensorflow.keras import layers


def get_model(gene_list, input_shape=(28, 28, 1), num_classes=10):
    gene_mapping = {
        -1: lambda: layers.Lambda(tf.identity),
        0: lambda: layers.Conv2D(32, kernel_size=3, activation="relu"),
        1: lambda: layers.BatchNormalization(),
        2: lambda: layers.MaxPool2D()
    }
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for gene in gene_list:
        x = gene_mapping[gene]()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def get_estimator(gene, batch_size=32, epochs=50):
    try:
        train_data, eval_data = cifar10.load_data()
        pipeline = fe.Pipeline(train_data=train_data,
                               eval_data=eval_data,
                               batch_size=batch_size,
                               ops=Minmax(inputs="x", outputs="x"))
        model = fe.build(model_fn=lambda: get_model(gene, input_shape=(32, 32, 3)), optimizer_fn="adam")
        network = fe.Network(ops=[
            ModelOp(model=model, inputs="x", outputs="y_pred"),
            CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
            UpdateOp(model=model, loss_name="ce")
        ])
        # step 3
        traces = [Accuracy(true_key="y", pred_key="y_pred")]
        estimator = fe.Estimator(pipeline=pipeline, network=network, epochs=epochs, traces=traces)
    except:
        estimator = 0
    return estimator


def get_random_gene_list(choices=[-1, 0, 1, 2], max_size=10):
    return [random.choice(choices) for _ in range(max_size)]


def mutate_gene(gene_list, choices=[-1, 0, 1, 2]):
    new_gene_list = [i for i in gene_list]
    mutate_idx = random.choice(list(range(len(new_gene_list))))
    new_gene_list[mutate_idx] = random.choice(choices)
    return new_gene_list


def evaluate_gene(gene, epochs=30, batch_size=128):
    est = get_estimator(gene=gene, epochs=epochs, batch_size=batch_size)
    if est == 0:
        best_acc = 0.0
    else:
        summary = est.fit(summary="exp")
        best_acc = float(max(summary.history["eval"]["accuracy"].values()))
    return best_acc


# def evaluate_gene2(gene):
#     # for debugging purpose
#     return sum(gene)


def regularized_evolution(num_generations=550, population_sizes=150, sample_size=25):
    history = {}
    results_history = []
    populations = [get_random_gene_list() for _ in range(population_sizes)]
    for _ in range(num_generations):
        tournament_pool = random.sample(populations, sample_size)
        tournament_results = []
        for gene in tournament_pool:
            if not tuple(gene) in history:
                history[tuple(gene)] = evaluate_gene(gene)
            tournament_results.append((tuple(gene), history[tuple(gene)]))
        results_history.append([x[1] for x in tournament_results])
        best_gene = max(tournament_results, key=lambda x: x[1])[0]
        best_gene = mutate_gene(best_gene)
        populations.insert(0, best_gene)
        populations.pop()
    results_history = np.array(results_history)
    np.save("results_history.npy", results_history)
    all_time_best_gene, performance = max(history.items(), key=lambda x: x[1])
    print("evolution finished, best gene is {}, with performance {}".format(all_time_best_gene, performance))


if __name__ == "__main__":
    regularized_evolution()
