"""
Example program creating a synthetic dataset and then using SMOTE and extensions of SMOTE from the imblearn library to
oversample the data and create a more balanced dataset.
"""

from collections import Counter
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from diffDists import plot_dataset
from randOverUnd import eval_pipeline


# Create method to transform dataset using a given method & output a scatter plot
def transformed_data(method, x, y):
    # Transform dataset
    oversample = method()
    newX, newY = oversample.fit_resample(x, y)
    # Summarize new class distribution
    new_counter = Counter(newY)
    print(new_counter)
    # Scatter plot of examples based on class label
    plot_dataset(newX, newY)


# Define imbalanced classification dataset
x, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                           flip_y=0, random_state=1)
# Summarize class distribution
counter = Counter(y)
print(counter)
plot_dataset(x, y)
model = DecisionTreeClassifier()
# Evaluate base level accuracy
eval_pipeline(model, x, y, 'roc_auc')

# Transform dataset using SMOTE
transformed_data(SMOTE, x, y)
# Define SMOTE pipeline
smote_steps = [('over', SMOTE()), ('model', DecisionTreeClassifier())]
smote_pipeline = Pipeline(steps=smote_steps)
eval_pipeline(smote_pipeline, x, y, 'roc_auc')

# Can do both over & under sampling
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
both_steps = [('o', over), ('u', under), ('m', DecisionTreeClassifier())]
both_pipeline = Pipeline(steps=both_steps)
eval_pipeline(both_pipeline, x, y, 'roc_auc')
# Transform dataset
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
new_x, new_y = pipeline.fit_resample(x, y)
# Summarize and plot new dataset
new_counter = Counter(new_y)
print(new_counter)
plot_dataset(new_x, new_y)

# Evaluate different values of k for SMOTE
k_values = [1, 2, 3, 4, 5, 6, 7]
for k in k_values:
    over_k = SMOTE(sampling_strategy=0.1, k_neighbors=k)
    under_k = RandomUnderSampler(sampling_strategy=0.5)
    k_steps = [('over', over_k), ('under', under_k), ('model', DecisionTreeClassifier())]
    k_pipeline = Pipeline(steps=k_steps)
    # Evaluate k pipelines
    eval_pipeline(k_pipeline, x, y, 'roc_auc', k=k)

# Transform data using borderline-SMOTE
transformed_data(BorderlineSMOTE, x, y)

# Transform data using borderline-SMOTE with SVM
transformed_data(SVMSMOTE, x, y)

# Transform data using ADASYN
transformed_data(ADASYN, x, y)


