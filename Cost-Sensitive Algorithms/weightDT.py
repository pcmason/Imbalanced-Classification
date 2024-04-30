"""
Example code creating a synthetic 1:100 imbalanced classification dataset, getting a baseline classification decision tree
accuracy & improving it with a weighted Decision Tree. Will use weights 1st based on the inverse class heuristic,
then will use grid-search to determine accuracy for multiple different weightings of the classes.
"""
from weightLR import *
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    # Generate dataset
    x, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=3)
    # Define model (weights based on inverse class)
    model = DecisionTreeClassifier(class_weight='balanced')
    # Evaluate classification decision tree
    model_evaluation(model, x, y)
    # Create a model that is not already weighted and use grid search to find best weighting
    nw_model = DecisionTreeClassifier()
    # Grid search the model
    grid_search_evaluation(nw_model, x, y)