'''
Example program using a synthetic binary classification dataset that has a 1:100 class imbalance and showing how
classification accuracy does not perform much better than a naive model that only picks the majority class.
'''
from diffDists import get_dataset
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
import numpy as np


# Method to evaluate a model using repeated k-fold CV
def evaluate_model(x, y, metric):
    # Define model as dummy classifier
    model = DummyClassifier(strategy='most_frequent')
    # Evaluate model with repeated stratified k-fold cv to ensure equal class distribution for each fold
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, x, y, scoring=metric, cv=cv, n_jobs=-1)
    return scores


# Define a 1:100 class distribution
proportions = {0:10000, 1:100}
# Generate dataset
x, y = get_dataset(proportions)
# Summarize class distribution
major = (len(np.where(y == 0)[0]) / len(x)) * 100
minor = (len(np.where(y == 1)[0]) / len(x)) * 100
print('Class 0: %.3f%%, Class 1: %.3f%%' % (major, minor))
# Evaluate naive model
scores = evaluate_model(x, y, 'accuracy')
# Report accuracy score of naive model
print('Accuracy: %.3f%%' % (np.mean(scores) * 100))
