"""
Example code creating a synthetic 1:100 imbalanced classification dataset, getting a baseline extreme gradient boosting
accuracy & improving it with a weighted XGBoost. Will use weights 1st based on the inverse class heuristic,
then will use grid-search to determine accuracy for multiple different weightings of the classes.
"""
from weightLR import model_evaluation
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from xgboost import XGBClassifier

def grid_search_evaluation(model, x, y):
    weights = [1, 10, 25, 50, 75, 99, 100, 1000]
    param_grid = dict(scale_pos_weight=weights)
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
    # Generate dataset
    x, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=7)
    # Define model (weights based on inverse class)
    model = XGBClassifier(scale_pos_weight=99)
    # Evaluate classification decision tree
    model_evaluation(model, x, y)
    # Create new XGB classifier that is not already weighted
    nw_model = XGBClassifier()
    # Grid search the model
    grid_search_evaluation(nw_model, x, y)