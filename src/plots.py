"""
Ploting functions
"""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# Use the seborn style
plt.style.use("seaborn")
# But with fonts from the document body
plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "font.size": 16,
    }
)


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


# Plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, class_names, title):
    """Plot the confusion matrix."""
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_tsne(
    features,
    labels,
    title="t-SNE plot",
    perplexity=30,
    n_iter=1000,
    palette="Paired",
    exclude_labels=None,
):
    """
    Apply t-SNE on the input features and plot the result using seaborn with the option to exclude specific labels.

    Args:
        features (numpy.ndarray): The feature array for the data.
        labels (numpy.ndarray): The labels corresponding to the features.
        title (str): The title of the plot.
        perplexity (int): The perplexity parameter for t-SNE. Defaults to 30.
        n_iter (int): Number of iterations for optimization. Defaults to 1000.
        palette (str): Color palette for seaborn. Defaults to 'Set1'.
        exclude_labels (list): A list of labels to exclude from the plot. Defaults to None.

    Returns:
        None: The function plots the t-SNE results.
    """
    print("Applying t-SNE...")
    # Define the mapping from numerical labels to descriptive labels
    # label_mapping = {0: "Backspin", 1: "No Spin", 2: "Topspin"}
    # Apply the mapping to convert numerical labels to descriptive labels
    # labels = np.vectorize(label_mapping.get)(labels)
    # Convert the features and labels into a DataFrame for filtering
    df = pd.DataFrame(features)
    df["label"] = labels

    # Exclude specific labels if provided
    if exclude_labels is not None:
        print("Excluding some labels")
        df = df[~df["label"].isin(exclude_labels)]

    # Extract the features and labels after filtering
    filtered_features = df.drop("label", axis=1).values
    filtered_labels = df["label"].values

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=0)
    tsne_results = tsne.fit_transform(filtered_features)

    # Create a dataframe for the t-SNE results
    tsne_df = pd.DataFrame(
        {
            "Component 1": tsne_results[:, 0],
            "Component 2": tsne_results[:, 1],
            "label": filtered_labels,
        }
    )
    # print(set(filtered_labels))

    # Create the plot
    plt.figure(figsize=(3, 3))
    ax = sns.scatterplot(
        x="Component 1",
        y="Component 2",
        hue="label",
        palette=palette,
        data=tsne_df,
        alpha=0.7,
        legend="auto",
        s=35,
    )
    ax.set(xlabel=None)
    ax.set(ylabel=None)

    # handles, labels = ax.get_legend_handles_labels()
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    # print(specgram)
    ax.imshow(
        specgram,
        # librosa.power_to_db(specgram),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="CMRmap",
    )
    ax.grid(False)


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
