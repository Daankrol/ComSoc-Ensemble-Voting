import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
from prepare_dataset import Dataset
from classifier import Classifier


class NeuralNet(Classifier):
    def __init__(self, n_classes=26, min_nodes=16, max_nodes=128, epochs=100) -> None:
        super().__init__()
        self.n_layers = random.choice([2, 3, 4])
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.epochs = epochs
        self.n_classes = n_classes
        self.n_nodes = []
        self.activations = []
        for l in range(self.n_layers):
            # uniformly chose number of nodes for each layer
            x = random.uniform(np.log(self.min_nodes), np.log(self.max_nodes))
            self.n_nodes.append(int(np.exp(x)))
            # chose activation functions
            self.activations.append(random.choice(['relu', 'tanh']))
        self.build_model(n_classes)

    def set_dataset(self, x_train, y_train, x_test, y_test):
        super().set_dataset(x_train, y_train, x_test, y_test)
        # compute batch size
        x = len(self.y_train) / 100
        self.batch_size = int(2 ** (np.ceil(np.log2(x))))

    def build_model(self, n_classes) -> None:
        input_l = keras.Input(shape=(16,))
        for l in range(self.n_layers):
            x = layers.Dense(self.n_nodes[l], activation=self.activations[l])(
                input_l if l == 0 else x)
        x = layers.Dense(n_classes, activation='softmax')(x)
        self.model = keras.Model(input_l, x)

    def fit(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='categorical_accuracy', patience=5
        )
        self.model.compile(loss=tf.losses.CategoricalCrossentropy(),
                           optimizer=tf.optimizers.RMSprop(),
                           metrics=[tf.metrics.CategoricalAccuracy(), tfa.metrics.F1Score(
                               num_classes=self.n_classes, average='weighted')],
                           )
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[early_stopping],
            verbose=0
        )
        self.history = history

    def predict(self, x):
        return self.model.predict(x)

    def f1_score(self):
        evaluation = self.model.evaluate(
            self.x_test, self.y_test, verbose=0)
        f1 = dict(zip(self.model.metrics_names, evaluation))['f1_score']
        return f1

    def __str__(self) -> str:
        return f'\nNN: \nlayers:{self.n_layers} \nnodes: {self.n_nodes} \
            \nactivations: {self.activations} \nbatch size: {self.batch_size} \nf1: {self.f1_score()}'


def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))
