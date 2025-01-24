import pandas as pd
import numpy as np

def load_digit_recognizer(train_csv_path, test_csv_path, train_val_percentage_split = 0.8):
    """
    Loads and preprocesses the digit recognizer dataset, splitting it into training, validation, and test sets.

    Args:
        train_csv_path (str): Path to the CSV file containing the training data, including labels.
        test_csv_path (str): Path to the CSV file containing the test data (no labels).
        train_val_percentage_split (float, optional): The fraction of the training data to use as the training set.
            The remaining fraction will be used for validation. Defaults to 0.8.

    Returns:
        A tuple containing:
        - train_data (np.ndarray): Training set features, normalized to the range [0, 1].
        - train_labels (np.ndarray): Labels corresponding to the training set.
        - val_data (np.ndarray): Validation set features, normalized to the range [0, 1].
        - val_labels (np.ndarray): Labels corresponding to the validation set.
        - test_data (np.ndarray): Test set features, normalized to the range [0, 1].

    Notes:
        - The training data is shuffled before splitting into training and validation sets.
        - Normalization is applied to the input features (not the labels), scaling the pixel values
          from the range [0, 255] to [0, 1].
    """
    # read each file with pandas
    data_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    # convert the dataframe to numpy arrays
    data_np = np.array(data_df, dtype=float)
    test_np = np.array(test_df, dtype=float)
    m, n = data_np.shape
    # shuffling the data before splitting into train and validation sets
    np.random.shuffle(data_np)
    # normalizing from 0-255 to 0-1 range only the inputs and not the labels
    data_np[:, 1:] = data_np[:, 1:] / 255.0
    test_np = test_np / 255.0
    # splitting the data into train and validation sets
    train_percentage = train_val_percentage_split
    train_np = data_np[: int(train_percentage*m)]
    val_np = data_np[int(train_percentage*m) : ]
    # splitting the input data and the labels
    train_data = train_np[:, 1:]
    val_data = val_np[:, 1:]
    test_data = test_np
    train_labels = train_np[:, 0]
    val_labels = val_np[:, 0]

    return (
        train_data, train_labels, 
        val_data, val_labels, 
        test_data
    )