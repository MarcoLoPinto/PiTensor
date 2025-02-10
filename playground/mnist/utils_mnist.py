import os, sys, csv
from collections import defaultdict
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union

# Import the necessary layers, losses, optimizers, and metrics:
from pitensor.nn.layers import Sequential
from pitensor.nn.losses import CrossEntropyLoss
from pitensor.nn.optimizers import Optimizer, SGD
from pitensor.metrics import precision_score, recall_score, f1_score
# Use the data loader to load the MNIST dataset:
from playground.data_loaders.digit_recognizer import load_digit_recognizer
# Import the necessary modules to plot the images:
import matplotlib.pyplot as plt

class ClassificationNetwork:
    """A simple sequential neural network for classification tasks.
    """
    def __init__(self, layers: Sequential):
        self.layers = layers
        self.loss = CrossEntropyLoss()

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
            save_dir_path: str,
            min_f1_score: float = 0.95
        ):
        num_batches = len(train_data) // batch_size

        best_f1 = 0.0
        # Save training history to a CSV file:
        history_path = os.path.join(save_dir_path, 'history.csv')
        save_path = os.path.join(save_dir_path, 'best.npy')
        os.makedirs(save_dir_path, exist_ok=True)
        with open(history_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train_loss', 'val_loss', 'precision', 'recall', 'f1'])

        print(f"Training started for {epochs} epochs...")

        for epoch in range(epochs):

            # Train epoch:

            train_loss = 0.0
            epoch_run = tqdm(range(num_batches))
            for i in epoch_run:

                batch_data = train_data[i*batch_size : (i + 1)*batch_size]
                batch_labels = train_labels[i*batch_size : (i + 1)*batch_size].astype(np.int64)

                predictions, loss = self.train_step(batch_data, batch_labels, optimizer)
                train_loss += loss

                epoch_run.set_description(f"Epoch {epoch + 1}/{epochs} [loss = {loss:.4f}]")

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
            
            # Append to CSV file
            with open(history_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([train_loss, val_loss, precision, recall, f1])

            if f1 > best_f1:
                best_f1 = f1
                if best_f1 > min_f1_score:
                    self.save_parameters(save_path)
                    print(f"Best model saved with F1: {f1:.4f}")
            
        print(f"Training completed!")

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
        predictions = self.layers.forward(input)
        return np.argmax(predictions, axis=1)

# region plots

def plot_grayscale_image(image_array, title=''):
    # Reshape the array to 28x28
    reshaped_image = image_array.reshape((28, 28))
    # Plot the image
    plt.title(title)
    plt.imshow(reshaped_image, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.show()

def plot_training_history(csv_path: str, save_path: str = None):
    """
    Reads a training history CSV file and plots training loss, validation loss, 
    and F1-score over epochs (without relying on 'epoch' column).

    Args:
        csv_path (str): Path to the CSV file containing training history.
    """
    # Read CSV file in one line (excluding header)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader) # Read the header row
        data = np.array([list(map(float, row)) for row in reader]) # Convert rows to NumPy array

    # Find column indices dynamically
    col_idx = {col: headers.index(col) for col in ["train_loss", "val_loss", "precision", "recall", "f1"]}

    # Extract training history
    train_loss, val_loss = data[:, col_idx["train_loss"]], data[:, col_idx["val_loss"]]
    precision, recall, f1_scores = data[:, col_idx["precision"]], data[:, col_idx["recall"]], data[:, col_idx["f1"]]

    # Plot Loss Curves
    plt.figure(figsize=(12, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Subplot 2: F1-Score
    plt.subplot(1, 2, 2)
    plt.plot(precision, label="Precision")
    plt.plot(recall, label="Recall")
    plt.plot(f1_scores, label="F1-score")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Metrics Over Epochs")
    plt.legend()
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# endregion