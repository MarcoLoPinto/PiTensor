import os, sys
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from pitensor.nn.layers import Linear, ReLU, Softmax, Sequential
from pitensor.nn.losses import CrossEntropyLoss
from pitensor.nn.optimizers import Optimizer, SGD
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Union

from playground.data_loaders.digit_recognizer import load_digit_recognizer

class Network:
    def __init__(self):
        self.layers = Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 10),
            Softmax()
        )
        self.loss = CrossEntropyLoss()
        self.best_f1 = 0

    def backward(self):
        grad = self.loss.backward()
        return self.layers.backward(grad)

    def train_step(self, input: np.ndarray, targets: np.ndarray, optimizer: Union[Optimizer, None] = None):
        predictions = self.layers.forward(input)
        loss = self.loss.forward(predictions, targets)
        if optimizer is not None:
            self.backward()
            optimizer.step(self.layers)
        return predictions, loss

    def evaluate_metrics(self, predicted_classes, targets):
        precision = precision_score(targets, predicted_classes, average='macro')
        recall = recall_score(targets, predicted_classes, average='macro')
        f1 = f1_score(targets, predicted_classes, average='macro')
        return precision, recall, f1

    def train(
            self, 
            train_data: np.ndarray, train_labels: np.ndarray, 
            val_data: np.ndarray, val_labels: np.ndarray, 
            epochs: int, batch_size: int, optimizer: Optimizer, 
            save_path: str = "best_model.npy",
        ):

        num_batches = len(train_data) // batch_size

        for epoch in range(epochs):

            # Train epoch:

            train_loss = 0.0
            for i in range(num_batches):

                batch_data = train_data[i*batch_size : (i + 1)*batch_size]
                batch_labels = train_labels[i*batch_size : (i + 1)*batch_size].astype(np.int64)

                predictions, loss = self.train_step(batch_data, batch_labels, optimizer)
                train_loss += loss

            train_loss /= num_batches

            # Validation:
            
            val_predictions = []
            val_targets = []

            val_batches = len(val_data) // batch_size
            val_loss = 0.0
            for i in range(val_batches):
                
                batch_data = val_data[i * batch_size:(i + 1) * batch_size]
                batch_labels = val_labels[i * batch_size:(i + 1) * batch_size].astype(np.int64)

                predictions, loss = self.train_step(batch_data, batch_labels, optimizer = None)
                
                val_predictions.append(np.argmax(predictions, axis=1))
                val_loss += loss
                
                val_targets.append(batch_labels)

            val_loss /= val_batches
            val_predictions = np.concatenate(val_predictions)
            val_targets = np.concatenate(val_targets)

            precision, recall, f1 = self.evaluate_metrics(val_predictions, val_targets)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            if f1 > self.best_f1:
                self.best_f1 = f1
                self.save_parameters(save_path)
                print(f"Best model saved with F1: {f1:.4f}")

    def save_parameters(self, file_path):
        params = dict(
            layers = self.layers.get_parameters(),
        )
        np.save(file_path, params, allow_pickle=True)
        print(f"Model parameters saved to {file_path}")

    def load_parameters(self, file_path):
        try:
            params = np.load(file_path, allow_pickle=True).item()
            for (layer_idx, layer), layer_params in zip(enumerate(self.layers), params["layers"]):
                self.layers[layer_idx].update_parameters(layer_params)
            print(f"Model parameters loaded from {file_path}")
        except Exception as e:
            print(f"Error loading model parameters: {e}")

    def predict(self, input):
        predictions = self.forward(input)
        return np.argmax(predictions, axis=1)

# SET THE SEED
np.random.seed(1746911)

# Load the MNIST dataset
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DIGIT_RECOGNIZER_DIR = os.path.join(ROOT_DIR, 'data', 'digit-recognizer')
TRAIN_FILE = os.path.join(DIGIT_RECOGNIZER_DIR, 'train.csv')
TEST_FILE = os.path.join(DIGIT_RECOGNIZER_DIR, 'test.csv')
CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'playground', 'checkpoints', 'mlp_mnist.npy')

train_data, train_labels, val_data, val_labels, test_data = load_digit_recognizer(
    TRAIN_FILE, TEST_FILE, train_val_percentage_split = 0.8
)

pipeline_phase = 'train'

def plot_grayscale_image(image_array, title=''):
    # Reshape the array to 28x28
    reshaped_image = image_array.reshape((28, 28))
    # Plot the image
    plt.title(title)
    plt.imshow(reshaped_image, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.show()

if pipeline_phase == 'train':
    net = Network()
    optimizer = SGD(learning_rate=0.01)
    # net.load_parameters('best_model.npy')
    net.train(
        train_data, train_labels, 
        val_data, val_labels, 
        epochs = 300, batch_size = 64, 
        optimizer = optimizer,
        save_path = CHECKPOINT_PATH
    )

    num_tests = 10
    test_data_small = test_data[:num_tests]
    preds = net.predict(test_data_small)
    for test_i, pred_i in zip(test_data_small, preds):
        plot_grayscale_image(test_i, title=f'pred {pred_i}')

elif pipeline_phase == 'test':
    net = Network()
    net.load_parameters(CHECKPOINT_PATH)

    num_tests = 10
    test_data_small = test_data[:num_tests]
    preds = net.predict(test_data_small)
    for test_i, pred_i in zip(test_data_small, preds):
        plot_grayscale_image(test_i, title=f'pred {pred_i}')