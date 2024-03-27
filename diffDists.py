'''
Example code using the sklearn make_blobs() method to create sythetic binary classification problems with a 1;1, 1:10,
1:100 & 1:1000 distributions between the output variable (y).
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Create dataset with a given class distribution
def get_dataset(proportions):
    # Determine number of classes (always 2 here)
    n_classes = len(proportions)
    # Determine number of examples to generate for each class
    largest = max([v for k, v in proportions.items()])
    n_samples = largest * n_classes
    # Create dataset
    x, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=1, cluster_std=3)
    # Collect examples
    x_list, y_list = list(), list()
    for k, v in proportions.items():
        # Split based on class distribution
        row_ix = np.where(y == k)[0]
        selected = row_ix[:v]
        x_list.append(x[selected, :])
        y_list.append(y[selected])
    return np.vstack(x_list), np.hstack(y_list)


# Scatter plot of dataset, different colors for each class
def plot_dataset(x, y):
    # Create scatter plot for samples from each class
    n_classes = len(np.unique(y))
    for class_value in range(n_classes):
        # Get row indexes for samples with this class
        row_ix = np.where(y == class_value)[0]
        # Create scatter of these samples
        plt.scatter(x[row_ix, 0], x[row_ix, 1], label=str(class_value))
    # Show legend & plot
    plt.legend()
    plt.show()


# Define class distributions
proportions = [{0:5000, 1:5000}, {0:10000, 1:1000}, {0:10000, 1:100}, {0:10000, 1:10}]
for prop in proportions:
    # Generate datasets
    x, y = get_dataset(prop)
    # Plot datasets
    plot_dataset(x, y)