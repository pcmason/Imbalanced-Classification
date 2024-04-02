"""
Example code using random oversampling, undersampling & a combination of both on a synthetic imbalanced classification
dataset with a ratio of 1:1000. Examples used in pipelines to be re-usable
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# Create method for evaluating a pipeline using 10-fold CV
def eval_pipeline(pipeline, x, y, score_name, k=''):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, x, y, scoring=score_name, cv=cv, n_jobs=-1)
    score = np.mean(scores)
    print(f'{k} {score_name} Score: %.3f' % score)


if __name__ == '__main__':
    # Define synthetic imbalanced dataset
    x, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)

    # Define oversampling pipeline
    over_steps = [('over', RandomOverSampler()), ('model', DecisionTreeClassifier())]
    over_pipeline = Pipeline(steps=over_steps)
    # Evaluate oversampling pipeline
    eval_pipeline(over_pipeline, x, y, 'f1_micro')

    # Define undersampling pipeline
    under_steps = [('under', RandomUnderSampler()), ('model', DecisionTreeClassifier())]
    under_pipeline = Pipeline(steps=under_steps)
    # Evaluate undersampling piepline
    eval_pipeline(under_pipeline, x, y, 'f1_micro')

    # Combine under & over sampling
    # Change distribution from 1:100 to 1:10
    over = RandomOverSampler(sampling_strategy=0.1)
    # Change to 1:2
    under = RandomUnderSampler(sampling_strategy=0.5)
    # Define combination pipeline
    steps = [('o', over), ('u', under), ('m', DecisionTreeClassifier())]
    combo_pipeline = Pipeline(steps=steps)
    # Evaluate combo pipeline
    eval_pipeline(combo_pipeline, x, y, 'f1_micro')

