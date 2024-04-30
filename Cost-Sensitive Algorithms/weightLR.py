"""
Example code creating a synthetic 1:100 imbalanced classification dataset, getting a baseline logistic regression
accuracy & improving it with a weighted logistic regression. Will use weights 1st based on the inverse class heuristic,
then will use grid-search to determine accuracy for multiple different weightings of the classes.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression


# Method for defining CV method, evaluating the model & outputting the performance
def model_evaluation(model, x, y):
    # Define 10-fold CV
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Evaluate model & summarize performance
    scores = cross_val_score(model, x, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.3f' % np.mean(scores))


# Method to grid search different class weights and pick the best performing configuration
def grid_search_evaluation(model, x, y):
    balance = [{0: 100, 1: 1}, {0: 10, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 10}, {0: 1, 1: 100}]
    param_grid = dict(class_weight=balance)
    # Define 10 fold CV
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Define grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
    # Execute grid search
    grid_result = grid.fit(x, y)
    # Report best configuration
    print(f'Best: {grid_result.best_score_} using {grid_result.best_params_} ')
    # Report all configurations
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f'{mean} {stdev} with: {param}')


if __name__ == '__main__':
    # Generate an imbalanced dataset
    x, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=2)
    # Define logistic regression model
    model = LogisticRegression(solver='lbfgs')
    # Evaluate baseline model
    model_evaluation(model, x, y)

    # Now use a weight logistic regression model
    weight_model = LogisticRegression(solver='lbfgs', class_weight='balanced')
    # Evaluate weighted logistic regression model
    model_evaluation(weight_model, x, y)

    # Now use grid search to determine the best possible weighting group out of these options
    grid_search_evaluation(model, x, y)



