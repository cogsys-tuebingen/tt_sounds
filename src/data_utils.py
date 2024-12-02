# Data preparation function
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_balanced_spin_labels(features, spin_labels, n_per_label):
    """
    Balances the features and spin labels by sampling `n_per_label` examples from each label.

    Args:
        features (numpy.ndarray): Feature array.
        spin_labels (numpy.ndarray): Spin labels corresponding to the features.
        n_per_label (int): The number of samples per label.

    Returns:
        balanced_features (numpy.ndarray): Balanced feature array.
        balanced_spin_labels (numpy.ndarray): Balanced spin label array.
    """
    # Create a DataFrame to facilitate label filtering and sampling
    df = pd.DataFrame(features)
    df["spin_label"] = spin_labels

    # Get unique spin labels
    unique_labels_spin = np.unique(spin_labels)

    # Initialize lists to store sampled data
    sampled_features = []
    sampled_spin_labels = []

    for label in unique_labels_spin:
        # Filter data for the current label
        df_label = df[df["spin_label"] == label]
        n_samples = min(n_per_label, len(df_label))

        # Randomly sample the data if there are more samples than required
        if n_samples > 0:
            df_sampled = df_label.sample(n=n_samples, random_state=42)
            sampled_features.append(df_sampled.drop(["spin_label"], axis=1).values)
            sampled_spin_labels.append(df_sampled["spin_label"].values)

    # Concatenate all sampled data
    balanced_features = np.vstack(sampled_features)
    balanced_spin_labels = np.hstack(sampled_spin_labels)

    return balanced_features, balanced_spin_labels


def balance_labels(features, surface_labels, spin_labels, n_per_label):
    """
    Returns a balanced feature array and corresponding surface and spin labels with exactly `n_per_label` samples for each label.

    Args:
        features (numpy.ndarray): The feature array.
        surface_labels (numpy.ndarray): The surface labels.
        spin_labels (numpy.ndarray): The spin labels.
        n_per_label (int): The number of samples to return per label.

    Returns:
        balanced_features (numpy.ndarray): The balanced feature array.
        balanced_surface_labels (numpy.ndarray): The balanced surface labels.
        balanced_spin_labels (numpy.ndarray): The balanced spin labels.
    """
    # Create a DataFrame to facilitate label filtering and sampling
    df = pd.DataFrame(features)
    df["surface_label"] = surface_labels
    df["spin_label"] = spin_labels

    # Get unique labels
    unique_labels_surface = np.unique(surface_labels)
    unique_labels_spin = np.unique(spin_labels)

    # Initialize lists to store sampled data
    sampled_features = []
    sampled_surface_labels = []
    sampled_spin_labels = []

    for label in unique_labels_surface:
        # Filter data for the current label
        df_label = df[df["surface_label"] == label]
        n_samples = min(n_per_label, len(df_label))

        # Randomly sample the data if there are more samples than required
        if n_samples > 0:
            df_sampled = df_label.sample(n=n_samples, random_state=42)
            sampled_features.append(
                df_sampled.drop(["surface_label", "spin_label"], axis=1).values
            )
            sampled_surface_labels.append(df_sampled["surface_label"].values)
            sampled_spin_labels.append(df_sampled["spin_label"].values)

    # Concatenate all sampled data
    balanced_features = np.vstack(sampled_features)
    balanced_surface_labels = np.hstack(sampled_surface_labels)
    balanced_spin_labels = np.hstack(sampled_spin_labels)

    return balanced_features, balanced_surface_labels, balanced_spin_labels


def prepare_svm_data(dataset):
    """Prepare data for GMM classification with optimized memory usage."""
    num_samples = len(dataset)

    # Extract the shape of one sample to calculate the flattened size
    sample_shape = dataset[0][0].numpy().shape
    flattened_size = np.prod(sample_shape)

    # Pre-allocate arrays for efficiency
    features = np.zeros((num_samples, flattened_size))
    surface_labels = np.zeros(num_samples, dtype=int)
    spin_labels = np.zeros(num_samples, dtype=int)

    for i in tqdm(range(num_samples)):
        # Load spectrogram and labels
        sgram, surface_class, spin_class = dataset[i]

        # Flatten the spectrogram and store it in the pre-allocated array
        features[i] = sgram.numpy().flatten()

        # Store the corresponding labels
        surface_labels[i] = surface_class
        spin_labels[i] = spin_class

    return features, surface_labels, spin_labels
