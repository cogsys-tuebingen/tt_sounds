"""
Plot tsne
"""
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (balance_labels, get_balanced_spin_labels,
                        prepare_svm_data)
from dataset import SoundDS, spin_classes, surface_classes
from plots import plot_confusion_matrix, plot_tsne

if __name__ == "__main__":
    train_path = Path("../data/train.csv")
    test_path = Path("../data/test.csv")
    full_path = Path("../data/full.csv")
    data_path = Path("../data/sounds")

    # Load the dataset
    print("Loading dataset")
    # train_dataset = SoundDS(data_path, train_path)
    # test_dataset = SoundDS(data_path, test_path)
    full_dataset = SoundDS(data_path, full_path)
    # print(full_dataset)

    # Prepare data for GMM
    # print("Preparing train dataset")
    # (
    #     train_features,
    #     surface_train_labels,
    #     spin_train_labels,
    # ) = prepare_svm_data(train_dataset)
    # print("Preparing test dataset")
    # (
    #     test_features,
    #     surface_test_labels,
    #     spin_test_labels,
    # ) = prepare_svm_data(test_dataset)
    print("Preparing full dataset")
    (
        full_features,
        surface_full_labels,
        spin_full_labels,
    ) = prepare_svm_data(full_dataset)
    f = full_features[272 : 272 + 250]
    l = surface_full_labels[272 : 272 + 250]

    # Apply PCA preprocessing
    # print("Applying PCA preprocessing")
    # pca = PCA(n_components=50)  # Reducing to 50 components
    # train_features_pca = pca.fit_transform(train_features)
    # test_features_pca = pca.transform(test_features)
    # Assuming train_features, surface_train_labels, and spin_train_labels are your data

    plot_tsne(f, l)
    # balanced_features, balanced_surface_labels, balanced_spin_labels = balance_labels(
    #     train_features, surface_train_labels, spin_train_labels, n_per_label=100
    # )
    # plot_tsne(
    #     balanced_features,
    #     balanced_surface_labels,
    #     perplexity=30,
    #     exclude_labels=[11, 12, 13],
    # )
    # plot_tsne(train_features, surface_train_labels, exclude_labels=[11, 12, 13])
    # balanced_features, balanced_spin_labels = get_balanced_spin_labels(
    #     train_features, spin_train_labels, n_per_label=200
    # )
    # plot_tsne(balanced_features, balanced_spin_labels, palette="tab10")
    # plot_tsne(test_features, surface_test_labels, exclude_labels=[11, 12, 13])
