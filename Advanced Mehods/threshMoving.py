"""
Program that takes a synthetic imbalanced binary classification problem and decides the best threshold for a ROC Curve
example, Precision-Recall Curve example & one using grid search to determine the best threshold.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score

import matplotlib.pyplot as plt


# Method to locate the max of a list and output the best threshold & score
def largest_score(metric, thresh, scr):
    ix = np.argmax(metric)
    print(f'Best Threshold={thresh[ix]}, {scr}={metric[ix]}')
    return ix


# Method to plot the graph and select the best performing threshold
def plot_graph(x_var, y_var, x_label, y_label, ind):
    plt.plot(x_var, y_var, marker='.', label='Logistic')
    plt.scatter(x_var[ind], y_var[ind], marker='o', color='black', label='Best')
    # Axis labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


# Apply threshold to positive probabilities to create labels
def to_labels(pos_probs, thresh):
    return (pos_probs >= thresh).astype('int')


if __name__ == '__main__':
    # Base code used for all methods
    # Generate synthetic binary dataset
    x, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1,
                               weights=[0.99], flip_y=0, random_state=1)
    # Split into train & test splits
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.5, random_state=2, stratify=y)
    # Fit the logistic regression model
    model = LogisticRegression(solver='lbfgs')
    model.fit(train_x, train_y)
    # Predict probabilities
    yhat = model.predict_proba(test_x)
    # Keep probabilities for the minority outcome only
    yhat = yhat[:, 1]

    # Optimal Threshold for ROC Curve
    # Calculate ROC Curves
    fpr, tpr, thresholds = roc_curve(test_y, yhat)
    # Calculate G-Mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # Locate the index of the largest G-Mean
    best_ind = largest_score(gmeans, thresholds, 'G-Mean')
    # Plot the ROC Curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plot_graph(fpr, tpr, 'False Positive Rate', 'True Positive Rate', best_ind)

    # Can get the best threshold using the Youden's J Statistic
    j = tpr - fpr
    jx = np.argmax(j)
    best_thresh = thresholds[jx]
    print('Best Threshold [J Stat.]=%f, G-Mean=%.3f' % (best_thresh, gmeans[jx]))

    # Optimal Threshold for Precision-Recall Curve
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(test_y, yhat)
    # Convert to f scores
    f_score = (2 * precision * recall) / (precision + recall)
    # Locate index of largest f score
    best_fscore = largest_score(f_score, thresholds, 'F-Score')
    # Plot the precision-recall curve for the model
    no_skill = len(test_y[test_y == 1]) / len(test_y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plot_graph(recall, precision, 'Recall', 'Precision', best_fscore)

    # Optimal Threshold Tuning
    # Define thresholds [any objective function example]
    thresholds = np.arange(0, 1, 0.001)
    # Evaluate each threshold
    scores = [f1_score(test_y, to_labels(yhat, t)) for t in thresholds]
    # Get best threshold
    largest_score(scores, thresholds, 'F-Score')



