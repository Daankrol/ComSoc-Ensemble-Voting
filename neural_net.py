import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
from prepare_dataset import Dataset


class NeuralNet():
    def __init__(self, dataset: Dataset, min_nodes=16, max_nodes=128, epochs=100) -> None:
        self.n_layers = random.choice([2, 3, 4])
        self.dataset = dataset
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.epochs = epochs
        self.n_nodes = []
        self.activations = []
        for l in range(self.n_layers):
            # uniformly chose number of nodes for each layer
            x = random.uniform(np.log(self.min_nodes), np.log(self.max_nodes))
            self.n_nodes.append(int(np.exp(x)))
            # chose activation functions
            self.activations.append(random.choice(['relu', 'tanh']))

        # compute batch size
        x = len(self.dataset.train_labels) / 100
        self.batch_size = int(2 ** (np.ceil(np.log2(x))))

        self.build_model()
        self.history = self.compile_and_fit()
        self.evaluation = self.evaluate()
        print(dict(zip(self.model.metrics_names, self.evaluation)))

    def build_model(self) -> None:
        input_l = keras.Input(shape=(16,))
        for l in range(self.n_layers):
            x = layers.Dense(self.n_nodes[l], activation=self.activations[l])(
                input_l if l == 0 else x)
        x = layers.Dense(self.dataset.n_classes, activation='softmax')(x)
        self.model = keras.Model(input_l, x)

    def compile_and_fit(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5
        )
        self.model.compile(loss=tf.losses.CategoricalCrossentropy(),
                           optimizer=tf.optimizers.RMSprop(),
                           metrics=[tf.metrics.CategoricalAccuracy(), tfa.metrics.F1Score(
                               num_classes=self.dataset.n_classes)],
                           )
        ds = self.dataset
        print('Training model.')
        history = self.model.fit(
            ds.train_x,
            ds.train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(ds.val_x, ds.val_labels),
            callbacks=[early_stopping],
            verbose=0
        )
        return history

    def evaluate(self):
        print('\nEvaluation:')
        return self.model.evaluate(
            self.dataset.test_x, self.dataset.test_labels)

    def __str__(self) -> str:
        return f'NN: \nlayers:{self.n_layers} \nnodes: {self.n_nodes} \
            \nactivations: {self.activations} \nbatch size: {self.batch_size}'
