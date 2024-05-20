"""
This is a file that uses bagging, random forest and early ensemble methods that are intentionally adjusted to work on
imbalanced classification problems.
"""
import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier
# Does not ignore the FutureWarnings so is commented out
warnings.simplefilter(action='ignore', category=FutureWarning)


# Create method to create model, evaluate it using 10-fold cv & mean ROC AUC metric
def eval_model(method, x, y, weighting=''):
    # Define model
    if weighting != '':
        model = method(n_estimators=10, class_weight=weighting)
    else:
        model = method(n_estimators=10)
    # Define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Evaluate model & summarize performance
    scores = cross_val_score(model, x, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.3f' % np.mean(scores))


if __name__ == '__main__':
    # Generate synthetic imbalanced classification dataset
    x, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
    # Bagging
    # Now get baseline ROC AUC for standard bagging method
    eval_model(BaggingClassifier, x, y)
    # Now use bagging with random undersampling
    eval_model(BalancedBaggingClassifier, x, y)

    # Random Forest
    # Baseline random forest performance
    eval_model(RandomForestClassifier, x, y)
    # Random forest with class weighting
    eval_model(RandomForestClassifier, x, y, weighting='balanced')
    # Random forest with bootstrap class weighting
    eval_model(RandomForestClassifier, x, y, weighting='balanced_subsample')
    # Random forest with random undersampling
    eval_model(BalancedRandomForestClassifier, x, y)

    # Final method is using easy ensemble
    eval_model(EasyEnsembleClassifier, x, y)



