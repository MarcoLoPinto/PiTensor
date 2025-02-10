import os, sys, csv
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Literal, Union
# Import the necessary layers, losses, optimizers, and metrics:
from pitensor.nn.layers import Conv2D, Flatten, Linear, MaxPool2D, ReLU, Softmax, Sequential
from pitensor.nn.losses import CrossEntropyLoss
from pitensor.nn.optimizers import Optimizer, SGD
from pitensor.metrics import precision_score, recall_score, f1_score
# Use the data loader to load the MNIST dataset:
from playground.data_loaders.digit_recognizer import load_digit_recognizer
# Import the necessary modules to plot the images:
import matplotlib.pyplot as plt

from utils_mnist import ClassificationNetwork, plot_grayscale_image, plot_training_history

if __name__ == '__main__':
    # Variables:
    approach: Literal['mlp', 'cnn'] = 'cnn'
    pipeline_phase = 'train'

    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
    DIGIT_RECOGNIZER_DIR = os.path.join(ROOT_DIR, 'data', 'digit-recognizer')
    TRAIN_FILE = os.path.join(DIGIT_RECOGNIZER_DIR, 'train.csv')
    TEST_FILE = os.path.join(DIGIT_RECOGNIZER_DIR, 'test.csv')
    CHECKPOINT_DIR_PATH = os.path.join(ROOT_DIR, 'playground', 'checkpoints', f'{approach}_mnist')

    # SET THE SEED
    np.random.seed(1746911)

    # Load the MNIST dataset
    train_data, train_labels, val_data, val_labels, test_data = load_digit_recognizer(
        TRAIN_FILE, TEST_FILE, train_val_percentage_split = 0.8
    )

    if approach == 'mlp':
        layers = Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 10),
            Softmax()
        )
    elif approach == 'cnn':
        layers = Sequential(
            Conv2D(1, 8, 3),
            ReLU(),
            MaxPool2D(pool_size=(2, 2)),

            Conv2D(8, 16, 3),
            ReLU(),
            MaxPool2D(pool_size=(2, 2)),

            Flatten(),
            Linear(16 * 5 * 5, 128),
            ReLU(),
            Linear(128, 10),
            Softmax()
        )
        train_data = train_data.reshape(-1, 1, 28, 28)
        val_data = val_data.reshape(-1, 1, 28, 28)
        test_data = test_data.reshape(-1, 1, 28, 28)
    else:
        raise ValueError(f'Invalid approach: {approach}')

    if pipeline_phase == 'train':
        net = ClassificationNetwork(layers)
        optimizer = SGD(learning_rate=0.01)
        net.train(
            train_data, train_labels, 
            val_data, val_labels, 
            epochs = 300, 
            batch_size = 64, 
            optimizer = optimizer,
            save_dir_path = CHECKPOINT_DIR_PATH,
            min_f1_score = 0.95 if approach == 'mlp' else 0.85
        )

        num_tests = 10
        test_data_small = test_data[:num_tests]
        preds = net.predict(test_data_small)
        for test_i, pred_i in zip(test_data_small, preds):
            plot_grayscale_image(test_i, title=f'pred {pred_i}')

    elif pipeline_phase == 'test':
        net = ClassificationNetwork(layers)
        net.load_parameters(os.path.join(CHECKPOINT_DIR_PATH, 'best.npy'))

        plot_training_history(os.path.join(CHECKPOINT_DIR_PATH, 'history.csv'), save_path=os.path.join(CHECKPOINT_DIR_PATH, 'history.png'))

        num_tests = 10
        test_data_small = test_data[:num_tests]
        preds = net.predict(test_data_small)
        for test_i, pred_i in zip(test_data_small, preds):
            plot_grayscale_image(test_i, title=f'pred {pred_i}')