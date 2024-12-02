from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import prepare_svm_data
from dataset import SoundDS, spin_classes, surface_classes
from plots import plot_confusion_matrix


# Class for SVM classifier with separate models for surface and spin
class SVMClassifier:
    def __init__(self, kernel="linear", C=1.0):
        # SVM classifier for surface classification
        self.svm_surface = make_pipeline(
            StandardScaler(), SVC(kernel=kernel, C=C, probability=True, verbose=True)
        )
        # SVM classifier for spin classification
        self.svm_spin = make_pipeline(
            StandardScaler(), SVC(kernel=kernel, C=C, probability=True, verbose=True)
        )

    def fit_surface(self, X_surface, y_surface):
        """Fit the SVM for surface classification."""
        self.svm_surface.fit(X_surface, y_surface)

    def fit_spin(self, X_spin, y_spin):
        """Fit the SVM for spin classification."""
        self.svm_spin.fit(X_spin, y_spin)

    def predict_surface(self, X_surface):
        """Predict the surface labels."""
        return self.svm_surface.predict(X_surface)

    def predict_spin(self, X_spin):
        """Predict the spin labels."""
        return self.svm_spin.predict(X_spin)

    def predict_proba_surface(self, X_surface):
        """Predict the probabilities for surface labels."""
        return self.svm_surface.predict_proba(X_surface)

    def predict_proba_spin(self, X_spin):
        """Predict the probabilities for spin labels."""
        return self.svm_spin.predict_proba(X_spin)


# def save_svm_data(features, surface_labels, spin_labels, filename="svm_data.npz"):
#     """
#     Save the features and labels for SVM classification to a file in .npz format.

#     Args:
#         features (numpy.ndarray): The feature array for both surface and spin.
#         surface_labels (numpy.ndarray): The labels for the surface classifier.
#         spin_labels (numpy.ndarray): The labels for the spin classifier.
#         filename (str): The filename to save the data to. Defaults to 'svm_data.npz'.
#     """
#     # Save the data into an npz file
#     np.savez(
#         filename,
#         features=features,
#         surface_labels=surface_labels,
#         spin_labels=spin_labels,
#     )
#     print(f"Data saved successfully to {filename}")


if __name__ == "__main__":
    train_path = Path("../data/train.csv")
    test_path = Path("../data/test.csv")
    data_path = Path("../data/sounds")

    # Load the dataset
    print("Loading dataset")
    train_dataset = SoundDS(data_path, train_path)
    test_dataset = SoundDS(data_path, test_path)

    # Prepare data for SVM
    print("Preparing train dataset")
    (
        train_features,
        surface_train_labels,
        spin_train_labels,
    ) = prepare_svm_data(train_dataset)
    print("Saving train dataset")
    # save_svm_data(
    #     train_features, surface_train_labels, spin_train_labels, "train_svm_data.npz"
    # )
    print("Preparing test dataset")
    (
        test_features,
        surface_test_labels,
        spin_test_labels,
    ) = prepare_svm_data(test_dataset)
    print("Saving test dataset")
    # save_svm_data(
    #     test_features, surface_test_labels, spin_test_labels, "test_svm_data.npz"
    # )

    # print("Loading prepared data")
    # train_data = np.load("train_svm_data.npz")
    # train_features = train_data["features"]
    # surface_train_labels = train_data["surface_labels"]
    # spin_train_labels = train_data["spin_labels"]

    # test_data = np.load("test_svm_data.npz")
    # test_features = test_data["features"]
    # surface_test_labels = test_data["surface_labels"]
    # spin_test_labels = test_data["spin_labels"]

    # Create and train the SVM classifier
    svm_classifier = SVMClassifier(kernel="rbf", C=1.0)

    # Train surface classifier
    svm_classifier.fit_surface(train_features, surface_train_labels)

    # Evaluate the surface classifier
    surface_preds = svm_classifier.predict_surface(test_features)
    print("Surface Predictions:", surface_preds)

    print("\nSurface Classification Report:")
    print(
        classification_report(
            surface_test_labels, surface_preds, target_names=surface_classes
        )
    )
    print("Surface Accuracy:", accuracy_score(surface_test_labels, surface_preds))
    # Plot the confusion matrix for surface classification
    plot_confusion_matrix(
        surface_test_labels, surface_preds, surface_classes, "Surface Confusion Matrix"
    )

    # Train spin classifier
    svm_classifier.fit_spin(train_features, spin_train_labels)
    spin_preds = svm_classifier.predict_spin(test_features)
    print("Spin Predictions:", spin_preds)

    # Evaluate the spin classifier
    print("\nSpin Classification Report:")
    print(
        classification_report(spin_test_labels, spin_preds, target_names=spin_classes)
    )
    print("Spin Accuracy:", accuracy_score(spin_test_labels, spin_preds))
    # Plot the confusion matrix for spin classification
    plot_confusion_matrix(
        spin_test_labels, spin_preds, spin_classes, "Spin Confusion Matrix"
    )

    coefffs = svm_classifier.svm_surface.coef_
    img_coef = coeffs.reshape()

    # Optionally, if you want the probabilities for each class
    # surface_probs = svm_classifier.predict_proba_surface(surface_test_features)
    # spin_probs = svm_classifier.predict_proba_spin(spin_test_features)

    # print("Surface Probabilities:", surface_probs)
    # print("Spin Probabilities:", spin_probs)
