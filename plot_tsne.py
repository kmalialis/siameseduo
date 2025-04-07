import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne_two_dicts(dict1, dict2):
    """
    Processes two dictionaries with class indices as keys and deques of numpy arrays as values.
    Reduces the dimensionality of the arrays to 2D using t-SNE and plots the results in side-by-side subplots.
    Same class index is assigned the same color but with different symbols for each dictionary.

    Parameters:
        dict1, dict2 (dict): Two dictionaries with keys as class indices (0-9) and values as deques
                             containing numpy arrays.
    """
    markers = ['o', 's']  # Markers: circle for dict1, square for dict2
    colors = plt.cm.tab10.colors  # Color map for class indices (0-9)

    combined_data = []
    combined_labels = []
    dict_indicator = []

    # Process first dictionary
    for class_index, data_queue in dict1.items():
        for array in data_queue:
            if array[-1] == class_index:
                combined_data.append(array[:-1])
                combined_labels.append(class_index)
                dict_indicator.append(0)  # Marker for dict1
            else:
                raise ValueError(f"Data mismatch in dict1: Last value of array does not match key {class_index}")

    # Process second dictionary
    for class_index, data_queue in dict2.items():
        for array in data_queue:
            if array[-1] == class_index:
                combined_data.append(array[:-1])
                combined_labels.append(class_index)
                dict_indicator.append(1)  # Marker for dict2
            else:
                raise ValueError(f"Data mismatch in dict2: Last value of array does not match key {class_index}")

    # Convert data to numpy array
    combined_data = np.array(combined_data)
    combined_labels = np.array(combined_labels)
    dict_indicator = np.array(dict_indicator)

    # Summary statistics
    print("Summary Statistics:")
    print(f"- Number of classes in dict1: {len(dict1)}")
    print(f"- Number of elements per class in dict1: {[len(queue) for queue in dict1.values()]}")
    print(f"- Number of classes in dict2: {len(dict2)}")
    print(f"- Number of elements per class in dict2: {[len(queue) for queue in dict2.values()]}")
    print(f"- Feature dimension (excluding class index): {combined_data.shape[1]}")

    # Reduce dimensionality to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(combined_data)

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for dict_idx, (marker, ax) in enumerate(zip(markers, axes)):
        for class_index in range(10):
            class_mask = (combined_labels == class_index) & (dict_indicator == dict_idx)
            ax.scatter(
                reduced_data[class_mask, 0],
                reduced_data[class_mask, 1],
                label=f"Class {class_index}",
                c=[colors[class_index]],
                marker=marker,
                s=50,
                edgecolor='k'
            )
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight='bold')
        ax.set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight='bold')
        ax.legend(fontsize=12, title="Class", title_fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)

    plt.tight_layout()

    # Save the plot with a timestamped filename in the local directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./tsne_side_by_side_{timestamp}.png"
    # plt.savefig(filename)
    print(f"Side-by-side plot saved as: {filename}")
    plt.show()
